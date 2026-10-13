"""GPU and CPU-thread parity for intfloat/multilingual-e5-small against the accepted CPU fp32 fixtures.

Reads the protected `onnx-fp32-cpu` fixtures (read only): `embeddings.jsonl` (input ids and the
sentence-transformers PyTorch fp32 CPU embedding) and `tokenizer-batch.json` (one right-padded batch).
The fixtures do not store hidden states, so the hidden-state reference is recomputed here with the same
sentence-transformers modules in PyTorch fp32 on CPU, and its pooled output is checked against the
protected embeddings first.

Candidates, all fed the fixture ids (no tokenization):

- fp32 profile candidates: ONNX Runtime CUDA EP on `onnx/model.onnx` (TF32 off and ORT's default),
  PyTorch CUDA fp32, ONNX Runtime CPU EP at fixed intra-op thread counts.
- exploratory, not the reference (written to --exploratory-out): fp16 graph on the CUDA EP, PyTorch
  autocast fp16 and half weights, PyTorch with TF32 matmul, TensorRT EP if it loads.

For every ONNX Runtime session the node-to-provider assignment is read from an ORT profile, and
ORT's own placement log lines are captured, so CPU fallback is visible (plan section 11).

Usage (select the GPU with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>):
    python run_e5_cuda.py --model-dir <snapshot> --revision <sha> \
        --fixtures <protected/.../onnx-fp32-cpu> --out <dir> [--exploratory-out <dir>] \
        [--scratch <dir>] [--dump <dir>]
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from model_pipeline import fixtures  # noqa: E402

PAD_ID = 1
ONNX = "onnx/model.onnx"
PACKAGES = [
    "torch",
    "transformers",
    "sentence-transformers",
    "tokenizers",
    "onnxruntime-gpu",
    "onnx",
    "numpy",
    "nvidia-cudnn-cu13",
    "nvidia-cuda-runtime",
    "nvidia-cublas",
]
ASSETS = [
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "model.safetensors",
    ONNX,
]
CPU_THREADS = [1, 12]


def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean pooling over the attention mask, then L2 normalization, in float32 (same as run_e5.py)."""
    m = mask[..., None].astype(np.float32)
    summed = (hidden.astype(np.float32) * m).sum(axis=1)
    mean = summed / np.clip(m.sum(axis=1), 1e-9, None)
    return mean / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)


def summary(abs_values: np.ndarray) -> dict:
    a = np.asarray(abs_values, dtype=np.float64).ravel()
    if not a.size:
        return {"max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0}
    return {"max_abs": float(a.max()), "mean_abs": float(a.mean()), "p99_abs": float(np.quantile(a, 0.99))}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def padded(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(r) for r in rows)
    ids = np.array([r + [PAD_ID] * (width - len(r)) for r in rows], dtype=np.int64)
    mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=np.int64)
    return ids, mask


@contextlib.contextmanager
def captured_stderr():
    """Capture native stderr (ORT logs) into a list of lines."""
    lines: list[str] = []
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            yield lines
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            lines.extend(tmp.read().decode("utf-8", "replace").splitlines())


class OrtCandidate:
    def __init__(self, model_path: Path, providers: list, threads: int | None, profile_dir: Path):
        opts = ort.SessionOptions()
        opts.log_severity_level = 0
        if threads is not None:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        opts.enable_profiling = True
        opts.profile_file_prefix = str(profile_dir / "ort_profile")
        with captured_stderr() as log:
            self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.log = [
            line
            for line in log
            if "placed on" in line
            or "assigned to" in line
            or "Node(s)" in line
            or "Memcpy" in line
            or "fallback" in line.lower()
        ]
        self.providers = self.session.get_providers()
        self.provider_options = self.session.get_provider_options()
        self.threads = threads
        self._profiled = False
        self.placement: dict = {}
        self.run_options = ort.RunOptions()
        self.run_options.log_severity_level = 3

    def __call__(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        if self._profiled:
            return self.session.run(["last_hidden_state"], feeds, self.run_options)[0]
        with captured_stderr():
            out = self.session.run(["last_hidden_state"], feeds, self.run_options)[0]
            profile = self.session.end_profiling()
        self._profiled = True
        self.placement = placement_from_profile(Path(profile))
        return out


def placement_from_profile(path: Path) -> dict:
    events = json.loads(path.read_text())
    by_provider: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    seen = set()
    for ev in events:
        if ev.get("cat") != "Node" or not ev.get("name", "").endswith("_kernel_time"):
            continue
        args = ev.get("args", {})
        node = ev["name"][: -len("_kernel_time")]
        if node in seen:
            continue
        seen.add(node)
        by_provider[args.get("provider", "unknown")][args.get("op_name", "unknown")] += 1
    path.unlink(missing_ok=True)
    return {
        "source": "ORT profile of the first run (optimized graph, one entry per executed node)",
        "nodes_per_provider": {p: sum(c.values()) for p, c in sorted(by_provider.items())},
        "ops_per_provider": {p: dict(sorted(c.items())) for p, c in sorted(by_provider.items())},
    }


class TorchCandidate:
    def __init__(self, model: SentenceTransformer, device: str, mode: str):
        self.model, self.device, self.mode = model, device, mode

    def __call__(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        feats = {
            "input_ids": torch.from_numpy(ids).to(self.device),
            "attention_mask": torch.from_numpy(mask).to(self.device),
            "token_type_ids": torch.zeros(ids.shape, dtype=torch.int64, device=self.device),
        }
        with torch.inference_mode():
            if self.mode == "autocast-fp16":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.model(feats)["token_embeddings"]
            else:
                out = self.model(feats)["token_embeddings"]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        return out.float().cpu().numpy()


def evaluate(run, cases: list[dict], ref_hidden: dict, ref_emb: dict, batch_ids: list[str]) -> tuple[dict, dict]:
    hid, emb, per_case, outputs = [], [], [], {}
    for case in cases:
        ids = np.array([case["input_ids"]], dtype=np.int64)
        mask = np.ones_like(ids)
        h = run(ids, mask)
        e = pooled(h, mask)[0]
        outputs[case["id"]] = (h[0], e)
        dh = np.abs(h[0].astype(np.float64) - ref_hidden[case["id"]])
        de = np.abs(e.astype(np.float64) - ref_emb[case["id"]])
        hid.append(dh.ravel())
        emb.append(de)
        per_case.append(
            {
                "id": case["id"],
                "tokens": len(case["input_ids"]),
                "hidden_max_abs": float(dh.max()),
                "embedding_max_abs": float(de.max()),
                "cosine": cosine(e, ref_emb[case["id"]]),
                "nan_or_inf": bool(not np.isfinite(h).all()),
            }
        )
    by_id = {c["id"]: c for c in cases}
    b_ids, b_mask = padded([by_id[i]["input_ids"] for i in batch_ids])
    b_emb = pooled(run(b_ids, b_mask), b_mask)
    b_diff = np.abs(b_emb - np.stack([ref_emb[i] for i in batch_ids]))
    b_single = np.abs(b_emb - np.stack([outputs[i][1] for i in batch_ids]))
    b16, b16_self = [], []
    for start in range(0, len(cases), 16):
        chunk = cases[start : start + 16]
        ids, mask = padded([c["input_ids"] for c in chunk])
        e = pooled(run(ids, mask), mask)
        b16.append(np.abs(e - np.stack([ref_emb[c["id"]] for c in chunk])).ravel())
        b16_self.append(np.abs(e - np.stack([outputs[c["id"]][1] for c in chunk])).ravel())
    # Repeatability: the two longest cases run again on the same session.
    rerun = []
    for case in sorted(cases, key=lambda c: -len(c["input_ids"]))[:2]:
        ids = np.array([case["input_ids"]], dtype=np.int64)
        rerun.append(float(np.abs(run(ids, np.ones_like(ids)) - outputs[case["id"]][0][None]).max()))
    worst_h = max(per_case, key=lambda p: p["hidden_max_abs"])
    result = {
        "cases": len(cases),
        "last_hidden_state_vs_torch_cpu_fp32": summary(np.concatenate(hid)),
        "embedding_vs_protected": {**summary(np.concatenate(emb)), "min_cosine": min(p["cosine"] for p in per_case)},
        "padded_batch_fixture_vs_protected": summary(b_diff),
        "padded_batch_fixture_vs_own_single": summary(b_single),
        "padded_batch16_all_cases_vs_protected": summary(np.concatenate(b16)),
        "padded_batch16_all_cases_vs_own_single": summary(np.concatenate(b16_self)),
        "rerun_same_session_max_abs_hidden": max(rerun),
        "worst_hidden_case": {"id": worst_h["id"], "tokens": worst_h["tokens"], "max_abs": worst_h["hidden_max_abs"]},
        "nan_or_inf_cases": [p["id"] for p in per_case if p["nan_or_inf"]],
        "per_case": per_case,
    }
    return result, outputs


def gpu_environment() -> dict:
    env: dict = {}
    try:
        q = "name,driver_version,pci.bus_id,memory.total,clocks.max.sm,power.limit"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu=index,{q}", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout
        env["nvidia_smi"] = [line.strip() for line in out.splitlines() if line.strip()]
    except OSError:
        env["nvidia_smi"] = None
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER")
    env["torch_cuda"] = torch.version.cuda
    env["torch_cudnn"] = torch.backends.cudnn.version()
    env["torch_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    env["torch_capability"] = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    env["onnxruntime_build_info"] = ort.get_build_info()
    env["onnxruntime_cuda_version"] = getattr(ort, "cuda_version", None)
    return env


def cpu_flags() -> list[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags = line.split(":", 1)[1].split()
                return sorted(f for f in flags if f.startswith(("avx", "fma", "f16c", "amx")))
    except OSError:
        pass
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixtures", type=Path, required=True, help="protected onnx-fp32-cpu directory (read)")
    parser.add_argument("--out", type=Path, required=True, help="fp32 CUDA profile evidence")
    parser.add_argument("--exploratory-out", type=Path, help="fp16, TF32, TensorRT results (not the reference)")
    parser.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()) / "e5-cuda")
    parser.add_argument("--dump", type=Path, help="write per-candidate hidden states (npz) for cross-language checks")
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = fixtures.read_jsonl(args.fixtures / "embeddings.jsonl")
    batch_ids = [r["id"] for r in json.loads((args.fixtures / "tokenizer-batch.json").read_text())["rows"]]
    ref_emb = {c["id"]: np.array(c["embedding"], dtype=np.float32).astype(np.float64) for c in cases}
    onnx_path = args.model_dir / ONNX

    # Hidden-state reference: sentence-transformers modules, PyTorch fp32 CPU, one row per call.
    model = SentenceTransformer(str(args.model_dir), device="cpu")
    model.eval()
    ref_hidden, rerun_emb = {}, []
    with torch.inference_mode():
        for c in cases:
            ids = torch.tensor([c["input_ids"]])
            out = model({"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": ids * 0})
            ref_hidden[c["id"]] = out["token_embeddings"][0].numpy().astype(np.float64)
            rerun_emb.append(np.abs(out["sentence_embedding"][0].numpy() - ref_emb[c["id"]]).max())
    reference_check = {
        "description": "PyTorch fp32 CPU sentence-transformers embedding recomputed in this environment against "
        "the protected embeddings.jsonl; its hidden states are the hidden-state reference below",
        "torch_threads": torch.get_num_threads(),
        "embedding_max_abs": float(max(rerun_emb)),
    }
    print(f"reference recompute vs protected: {reference_check['embedding_max_abs']:.3g}")

    results: dict = {}
    placements: dict = {}
    dumps: dict = {}

    def record(name: str, description: str, run, sink: dict, keep: bool = False) -> None:
        res, outputs = evaluate(run, cases, ref_hidden, ref_emb, batch_ids)
        sink[name] = {"candidate": description, **res}
        if isinstance(run, OrtCandidate):
            placements[name] = {
                "providers": run.providers,
                "provider_options": run.provider_options,
                "intra_op_num_threads": run.threads,
                **run.placement,
                "ort_log_lines": run.log,
            }
        if keep and args.dump:
            dumps[name] = outputs
        h = res["last_hidden_state_vs_torch_cpu_fp32"]["max_abs"]
        e = res["embedding_vs_protected"]["max_abs"]
        print(f"{name}: hidden_max={h:.3g} emb_max={e:.3g} min_cos={res['embedding_vs_protected']['min_cosine']}")

    cuda = "CUDAExecutionProvider"
    record(
        "ort-cuda-fp32-tf32-off",
        "Python ORT CUDA EP, onnx/model.onnx fp32, use_tf32=0",
        OrtCandidate(onnx_path, [(cuda, {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"], None, args.scratch),
        results,
        keep=True,
    )
    record(
        "ort-cuda-fp32-default",
        "Python ORT CUDA EP, onnx/model.onnx fp32, provider defaults (use_tf32 as ORT ships it)",
        OrtCandidate(onnx_path, [(cuda, {"device_id": 0}), "CPUExecutionProvider"], None, args.scratch),
        results,
        keep=True,
    )
    for threads in CPU_THREADS:
        record(
            f"ort-cpu-fp32-threads{threads}",
            f"Python ORT CPU EP, onnx/model.onnx fp32, intra_op_num_threads={threads}, inter_op_num_threads=1",
            OrtCandidate(onnx_path, ["CPUExecutionProvider"], threads, args.scratch),
            results,
            keep=True,
        )
    model_cuda = SentenceTransformer(str(args.model_dir), device="cuda")
    model_cuda.eval()
    record(
        "torch-cuda-fp32",
        "PyTorch CUDA fp32 (sentence-transformers Transformer module), TF32 matmul off, deterministic algorithms",
        TorchCandidate(model_cuda, "cuda", "fp32"),
        results,
    )

    exploratory: dict = {}
    if args.exploratory_out:
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        record(
            "torch-cuda-tf32",
            "PyTorch CUDA, fp32 weights, TF32 matmul allowed",
            TorchCandidate(model_cuda, "cuda", "fp32"),
            exploratory,
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        record(
            "torch-cuda-autocast-fp16",
            "PyTorch CUDA, fp32 weights, torch.autocast float16",
            TorchCandidate(model_cuda, "cuda", "autocast-fp16"),
            exploratory,
        )
        model_half = SentenceTransformer(str(args.model_dir), device="cuda")
        model_half.half().eval()
        record(
            "torch-cuda-half",
            "PyTorch CUDA, weights and activations float16 (model.half())",
            TorchCandidate(model_half, "cuda", "half"),
            exploratory,
        )
        del model_half
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        fp16_path = args.scratch / "model_fp16.onnx"
        if not fp16_path.exists():
            converted = convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=True)
            onnx.save(converted, str(fp16_path))
        fp16_record = fixtures.file_record(fp16_path)
        fp16_record["recipe"] = "onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)"
        record(
            "ort-cuda-fp16-graph",
            "Python ORT CUDA EP, onnx/model.onnx converted to fp16 (fp32 inputs/outputs kept)",
            OrtCandidate(fp16_path, [(cuda, {"device_id": 0}), "CPUExecutionProvider"], None, args.scratch),
            exploratory,
        )
        exploratory["ort-cuda-fp16-graph"]["graph"] = fp16_record
        try:
            trt = OrtCandidate(
                onnx_path,
                [("TensorrtExecutionProvider", {"device_id": 0}), (cuda, {"device_id": 0}), "CPUExecutionProvider"],
                None,
                args.scratch,
            )
            if "TensorrtExecutionProvider" not in trt.providers:
                raise RuntimeError(f"TensorRT EP not active; session providers {trt.providers}")
            record("ort-tensorrt-fp32", "Python ORT TensorRT EP, fp32", trt, exploratory)
        except Exception as exc:  # noqa: BLE001 - recorded as evidence
            exploratory["ort-tensorrt-fp32"] = {"status": "untested", "error": str(exc)[:2000]}
            print(f"TensorRT EP: untested ({str(exc)[:200]})")

    env = fixtures.environment(PACKAGES)
    env.update(gpu_environment())
    env["cpu_isa_flags"] = cpu_flags()
    env["torch_threads"] = torch.get_num_threads()
    environment = {
        "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
        "runner": "model-pipeline/reference/encoder/run_e5_cuda.py",
        "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
        "model": {"repo": "intfloat/multilingual-e5-small", "revision": args.revision},
        "assets": [fixtures.file_record(args.model_dir / a, args.model_dir) for a in ASSETS],
        "reference_fixtures": [
            fixtures.file_record(args.fixtures / name) for name in ("embeddings.jsonl", "tokenizer-batch.json")
        ],
        "environment": env,
    }
    parity = {
        "reference": "protected onnx-fp32-cpu/embeddings.jsonl (sentence-transformers, PyTorch fp32 CPU); hidden "
        "states recomputed here with the same modules on CPU",
        "inputs": "input_ids from embeddings.jsonl (tokenizer.json ids, truncated to 512); attention_mask all 1 for "
        "single rows; token_type_ids all 0; padded batches right-padded with id 1",
        "pooling": "NumPy float32 mean over the attention mask, then L2 normalization",
        "reference_recompute": reference_check,
        "candidates": results,
    }
    fixtures.write_json(args.out / "parity.json", parity)
    fixtures.write_json(args.out / "placement.json", placements)
    fixtures.write_json(args.out / "environment.json", environment)
    if args.exploratory_out:
        fixtures.write_json(
            args.exploratory_out / "exploratory.json",
            {"note": "Not reference evidence; separate profile candidates.", "candidates": exploratory},
        )
        fixtures.write_json(args.exploratory_out / "environment.json", environment)
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for name, outputs in dumps.items():
            np.savez(
                args.dump / f"{name}.npz",
                **{f"hidden/{k}": v[0] for k, v in outputs.items()},
                **{f"embedding/{k}": v[1] for k, v in outputs.items()},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
