"""Matched-backend benchmark runner for Qwen3.5-0.8B on Python ONNX Runtime GenAI (G5, plan section 11).

The Python side of `ai-dotnet/benchmarks/Starlings.Benchmarks.Qwen35Harness`, for the budgets
`qwen3.5-0.8b/ortgenai-*` (`decode_tokens_per_second_ratio_min`, `time_to_first_token_ratio_max`) in
`model-pipeline/protected/performance-budgets.yaml`: Python `onnxruntime-genai` 0.17.1 (the C# side's
Microsoft.ML.OnnxRuntimeGenAI version) over `onnxruntime` 1.30.0, the same bundle, the same prompts (workload file
`qwen35-v1.json`, prompt ids of `model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl`), greedy, and
the same load and request settings as `GenAIModel.Load` and `TextGenerator`:

- `Config(bundle)`, `clear_providers()` (CPU) or `append_provider("cuda")` with `device_id` 0 and `use_tf32` 0,
  overlays `{"model":{"decoder":{"session_options":{"intra_op_num_threads":N}}}}` and
  `{"model":{"pad_token_id":248319}}` (Qwen35ChatModel.PadTokenIdOverlay), in that order;
- per request a fresh generator, `do_sample=False`, `max_length` = prompt + max_new_tokens; the prompt appended in
  one call on CPU and in chunks of 512 tokens on CUDA (Qwen35ChatOptions.DefaultCudaPrefillChunkTokens);
  `--prefill-chunk` overrides; stop at `<|endoftext|>` (248044) or `<|im_end|>` (248046) or max_new_tokens.

Stages and metrics per sample (as the C# harness): `template` (text path: chat template and tokenizer; 0 on the ids
path), `prefill` (the append calls), `decode` (every generated token including the stop token, with incremental
detokenization); `ttft_ms` (call start to the first streamed text, held back like the C# incremental detokenizer
while the text ends in an incomplete UTF-8 sequence), `prompt_tokens`, `completion_tokens`, `prefill_tokens_per_s`,
`decode_tokens_per_s` (decode steps, the stop token included, over decode time) and `inter_token_ms`.

The text path renders `chat_template.jinja` with Jinja2 as transformers does (sandboxed, `trim_blocks`,
`lstrip_blocks`) and tokenizes with Hugging Face `tokenizers`; the preliminary sweep checks the rendered text and ids
against the reference. Quality on every series: generated ids (with the stop token) equal to the reference output up
to max_new_tokens, prompt ids equal to the reference.

Usage:
    python bench_qwen35_matched.py warm --snapshot <snapshot> --provider cpu --threads 6 --out raw.jsonl
    python bench_qwen35_matched.py cold --snapshot <snapshot> --provider cpu --threads 6 --processes 10 --out raw.jsonl
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

HARNESS = "bench_qwen35_matched.py"
PROTECTED = core.PROTECTED / "qwen3.5-0.8b"
FIXTURE = PROTECTED / "reference" / "generation.jsonl"
PAD_OVERLAY = 248319
STOP_IDS = {248044, 248046}
CUDA_PREFILL_CHUNK = 512
GENAI_VERSION = "0.17.1"
STAGES = ["template", "prefill", "decode"]
METRICS = ["ttft_ms", "prompt_tokens", "completion_tokens", "prefill_tokens_per_s", "decode_tokens_per_s"]
METRICS += ["inter_token_ms"]


def chat_template(path: Path):
    """chat_template.jinja compiled as transformers compiles chat templates (sandbox, trim and lstrip blocks)."""
    from jinja2 import ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(path.read_text(encoding="utf-8"))


class QwenRun(core.LoadedRun):
    def __init__(self, args: argparse.Namespace):
        t0 = time.perf_counter_ns()
        import importlib.metadata

        import onnxruntime_genai as og
        import tokenizers

        self.og = og
        self.args = args
        version = importlib.metadata.version("onnxruntime-genai" + ("-cuda" if args.provider == "cuda" else ""))
        if version != GENAI_VERSION and not args.allow_genai_version:
            raise SystemExit(f"onnxruntime-genai {version}; the C# side runs {GENAI_VERSION} (--allow-genai-version)")
        stages = {"runtime_init": core.ms_since(t0)}
        t = time.perf_counter_ns()
        self.spec = json.loads(args.workloads.read_text(encoding="utf-8"))
        self.cases = {c["id"]: c for c in core.read_jsonl(FIXTURE)}
        stages["fixture_read"] = core.ms_since(t)
        t = time.perf_counter_ns()
        self.bundle = args.snapshot / f"ortgenai-{args.variant}"
        config = og.Config(str(self.bundle))
        config.clear_providers()
        self.provider_options = {}
        if args.provider == "cuda":
            config.append_provider("cuda")
            self.provider_options = {"device_id": "0", "use_tf32": "0"}
            for key, value in self.provider_options.items():
                config.set_provider_option("cuda", key, value)
        self.overlays = []
        if args.threads:
            session = {"intra_op_num_threads": args.threads}
            self.overlays.append(json.dumps({"model": {"decoder": {"session_options": session}}}))
        self.overlays.append(json.dumps({"model": {"pad_token_id": PAD_OVERLAY}}))
        for overlay in self.overlays:
            config.overlay(overlay)
        self.model = og.Model(config)
        stages["model_load"] = core.ms_since(t)
        t = time.perf_counter_ns()
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path(args.snapshot)))
        self.template = chat_template(args.snapshot / "chat_template.jinja")
        self.decoders = tokenizers.decoders
        stages["tokenizer_load"] = core.ms_since(t)
        self.load_stages = stages
        self.chunk = args.prefill_chunk or (CUDA_PREFILL_CHUNK if args.provider == "cuda" else None)
        self.settings = {
            "provider": args.provider,
            "intra_op_num_threads": args.threads,
            "inter_op_num_threads": None,
            "execution_mode": "onnxruntime-genai",
            "graph_optimization_level": "genai_config.json",
            "provider_options": self.provider_options,
            "genai_version": version,
            "config_overlays": self.overlays,
            "prefill_chunk_tokens": self.chunk,
            "sampling": "greedy",
            "variant": args.variant,
        }

    def render(self, case: dict) -> str:
        kwargs = case.get("chat_template_kwargs") or {}
        return self.template.render(
            messages=case["messages"], tools=case.get("tools"), add_generation_prompt=True, **kwargs
        )

    def generate(self, case: dict, text: bool, max_new: int):
        og = self.og
        t0 = time.perf_counter_ns()
        if text:
            ids = self.tokenizer.encode(self.render(case), add_special_tokens=False).ids
            template = core.ms_since(t0)
        else:
            ids, template = case["prompt_ids"], 0.0
        params = og.GeneratorParams(self.model)
        params.set_search_options(do_sample=False, max_length=len(ids) + max_new)
        generator = og.Generator(self.model, params)
        try:
            t_clock = time.perf_counter_ns()
            chunk = self.chunk or len(ids)
            for start in range(0, len(ids), chunk):
                generator.append_tokens(ids[start : start + chunk])
            prefill = core.ms_since(t_clock)
            stream = self.decoders.DecodeStream(skip_special_tokens=False)
            tokens, stop, ttft = [], None, -1.0
            while len(tokens) < max_new:
                if generator.is_done():
                    raise SystemExit(f"{case['id']}: GenAI ended the sequence after {len(tokens)} tokens")
                generator.generate_next_token()
                token = int(generator.get_next_tokens()[0])
                if token in STOP_IDS:
                    stop = token
                    break
                tokens.append(token)
                piece = stream.step(self.tokenizer, token)
                if piece and ttft < 0:
                    ttft = core.ms_since(t0)
            decode = core.ms_since(t_clock) - prefill
        finally:
            del generator
            del params
        if ttft < 0:  # the first update is the final result
            ttft = core.ms_since(t0)
        steps = len(tokens) + (1 if stop is not None else 0)
        stages = {"template": template, "prefill": prefill, "decode": decode}
        metrics = {
            "ttft_ms": ttft,
            "prompt_tokens": len(ids),
            "completion_tokens": len(tokens),
            "prefill_tokens_per_s": len(ids) / (prefill / 1000),
            "decode_tokens_per_s": steps / (decode / 1000) if steps else 0.0,
            "inter_token_ms": decode / steps if steps else 0.0,
        }
        return (
            {"case": case, "max_new": max_new, "prompt_ids": list(ids), "tokens": tokens, "stop": stop},
            stages,
            metrics,
        )

    @staticmethod
    def judge(outputs) -> dict:
        generations = equal = 0
        failures = []
        for _, x in outputs:
            generations += 1
            case = x["case"]
            expected = case["output_ids"][: x["max_new"]]
            actual = x["tokens"] + ([x["stop"]] if x["stop"] is not None else [])
            prompt_equal = x["prompt_ids"] == case["prompt_ids"]
            if prompt_equal and expected == actual:
                equal += 1
                continue
            n = 0
            while n < min(len(expected), len(actual)) and expected[n] == actual[n]:
                n += 1
            margin = case["step_margin"][n] if n < len(case["step_margin"]) else "n/a"
            failures.append(
                f"{case['id']}: diverges at step {n} (reference top-1 margin {margin})"
                if prompt_equal
                else f"{case['id']}: prompt ids differ"
            )
        return {
            "check": "greedy ids equal to the reference up to max_new_tokens (exact)",
            "generations": generations,
            "equal": equal,
            "failures": failures,
            "pass": equal == generations,
        }

    def case(self, case_id: str) -> dict:
        if case_id not in self.cases:
            raise SystemExit(f"Workload names unknown prompt '{case_id}'.")
        return self.cases[case_id]

    def series(self, name, path, input_, concurrency, samples, warmup, requests, max_new) -> core.Series:
        text = path == "text"
        if text and any(c.get("tools") for c in requests):
            raise SystemExit(f"{name}: prompts with tools run on the ids path only (as the C# harness)")
        return core.Series(
            workload=name,
            path=path,
            input=input_,
            batch=1,
            concurrency=concurrency,
            samples=samples,
            warmup_calls=warmup,
            requests=len(requests),
            stages=STAGES,
            metrics=METRICS,
            call=lambda r: self.generate(requests[r], text, max_new),
            judge=self.judge,
        )

    def warm_series(self):
        for w in self.spec["workloads"]:
            requests = [self.case(i) for i in w["requests"]]
            for c in w.get("concurrency", [1]):
                warmup = w.get("warmup_calls", self.spec["warmup_calls"])
                yield self.series(
                    w["name"], w["path"], w["input"], c, w["samples"], warmup, requests, w["max_new_tokens"]
                )

    def cold_series(self) -> core.Series:
        cold = self.spec["cold"]
        requests = [self.case(i) for i in cold["request"]]
        return self.series("cold-start", "ids", "short prompt", 1, 1, 0, requests, cold["max_new_tokens"])

    def preliminary_quality(self) -> dict:
        """Every prompt on the ids path, greedy for 16 tokens; the text path's rendering and ids (no tools)."""
        cases = list(self.cases.values())
        ids = self.judge([(i, self.generate(c, False, 16)[0]) for i, c in enumerate(cases)])
        template_cases = [c for c in cases if not c.get("tools")]
        exact = 0
        for c in template_cases:
            rendered = self.render(c)
            exact += rendered == c["rendered"] and (
                self.tokenizer.encode(rendered, add_special_tokens=False).ids == c["prompt_ids"]
            )
        return {
            "reference": core.repo_relative(FIXTURE),
            "cases": len(cases),
            "greedy_ids_16_tokens": ids,
            "template_cases": len(template_cases),
            "template_exact": exact,
            "pass": bool(ids["pass"] and exact == len(template_cases)),
        }


def tokenizer_path(snapshot: Path) -> Path:
    found = sorted(snapshot.glob("transformers-*/tokenizer.json"))
    if not found:
        raise SystemExit(f"no transformers-*/tokenizer.json under {snapshot}")
    return found[-1]


class QwenDriver(core.Driver):
    model = "qwen3.5-0.8b"
    harness = HARNESS
    impl = "python-genai"
    default_workloads = core.WORKLOADS / "qwen35-v1.json"
    snapshot_env = "STARLING_QWEN35_SNAPSHOT"

    def add_arguments(self, parser: argparse.ArgumentParser, command: str) -> None:
        parser.add_argument("--variant", choices=["fp32x", "fp16x"], default="fp32x")
        parser.add_argument("--prefill-chunk", type=int, default=None, help="default: one call on CPU, 512 on CUDA")
        parser.add_argument("--allow-genai-version", action="store_true", help="run another onnxruntime-genai version")

    def forwarded(self, args: argparse.Namespace) -> list[str]:
        argv = ["--variant", args.variant]
        if args.prefill_chunk:
            argv += ["--prefill-chunk", str(args.prefill_chunk)]
        if args.allow_genai_version:
            argv.append("--allow-genai-version")
        return argv

    def profile(self, args: argparse.Namespace) -> str:
        return f"ortgenai-{args.variant}-{args.provider}"

    def load(self, args: argparse.Namespace) -> core.LoadedRun:
        if args.snapshot is None:
            raise SystemExit("--snapshot or STARLING_QWEN35_SNAPSHOT is required")
        if args.variant == "fp16x" and args.provider == "cpu":
            raise SystemExit("fp16x is built for CUDA (as Qwen35ChatModel.Load)")
        return QwenRun(args)

    def reference(self, args: argparse.Namespace) -> str:
        return core.repo_relative(FIXTURE)

    def environment_extra(self, args: argparse.Namespace) -> dict:
        return {"native_libraries": core.loaded_native_libraries((*core.NATIVE_LIBRARIES, "onnxruntime-genai"))}

    def artifacts(self, args: argparse.Namespace) -> dict:
        bundle = args.snapshot / f"ortgenai-{args.variant}"
        dists = ("onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime", "onnxruntime-gpu", "numpy")
        return core.artifacts(
            bundle / "model.onnx",
            tokenizer_path(args.snapshot),
            args.workloads,
            core.file_digests(FIXTURE, PROTECTED / f"ortgenai-{args.variant}-{args.provider}" / "tolerances.yaml")
            if (PROTECTED / f"ortgenai-{args.variant}-{args.provider}" / "tolerances.yaml").exists()
            else core.file_digests(FIXTURE),
            core.package_versions((*dists, "tokenizers", "jinja2")),
            {
                "bundle_sha256": core.file_digests(
                    bundle / "model.onnx.data", bundle / "genai_config.json", args.snapshot / "chat_template.jinja"
                ),
            },
        )


def main() -> int:
    return core.main(QwenDriver(), Path(__file__).resolve(), _T_MAIN_NS, __doc__.split("\n", 1)[0])


if __name__ == "__main__":
    raise SystemExit(main())
