"""Stage 2 (export) results for multilingual-e5-small: Python ONNX Runtime against PyTorch (backlog P-007).

Runs `onnx/model.onnx` in Python ONNX Runtime on the protected fixture ids, and the
sentence-transformers PyTorch modules on the same ids to recompute reference hidden states.
Writes `export.results.json` in the stage results format (`model_pipeline.results`) plus two
float32 sidecars with the hidden states. It applies no thresholds; `model-pipeline evaluate`
does that against the profile's protected tolerances.

Usage (environment from requirements.txt, see reference/README.md):
    python stage_e5_export.py --model-dir <snapshot> --fixtures <protected onnx-fp32-cpu dir> \
        --profile multilingual-e5-small/onnx-fp32-cpu --out <results dir>

CUDA profile (environment from requirements-cuda.txt): add `--provider cuda` and pass
`--profile multilingual-e5-small/onnx-fp32-cuda`. ONNX Runtime then runs on the CUDA execution provider with
use_tf32=0; the PyTorch reference hidden states stay fp32 CPU.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer

from model_pipeline import fixtures, results

PAD_ID = 1
BATCH_SIZE = 16
PACKAGES = ["torch", "transformers", "sentence-transformers", "onnxruntime", "numpy"]


def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the attention mask, then L2 normalization, in float32 (as run_e5.py and sentence-transformers)."""
    m = mask[..., None].astype(np.float32)
    mean = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    return mean / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)


def ort_run(session: ort.InferenceSession, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    return session.run(["last_hidden_state"], feeds)[0]


def padded(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(r) for r in rows)
    ids = np.array([r + [PAD_ID] * (width - len(r)) for r in rows], dtype=np.int64)
    mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=np.int64)
    return ids, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--intra-op-threads", type=int, default=0, help="0 keeps the ONNX Runtime default")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    cases = fixtures.read_jsonl(args.fixtures / "embeddings.jsonl")
    batch_fx = json.loads((args.fixtures / "tokenizer-batch.json").read_text(encoding="utf-8"))

    model = SentenceTransformer(str(args.model_dir), device="cpu")
    model.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    model_path = args.model_dir / "onnx" / "model.onnx"
    provider_options: dict = {}
    if args.provider == "cuda":
        # The CUDA profiles pin use_tf32=0 (protected tolerances); there is no silent CPU fallback.
        provider_options = {"use_tf32": 0}
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    if args.provider == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise SystemExit(f"--provider cuda: the session has no CUDA execution provider ({session.get_providers()})")

    env = fixtures.environment(PACKAGES)
    env |= {
        "machine": platform.node(),
        "provider": args.provider,
        "provider_options": provider_options,
        "onnxruntime_providers": session.get_providers(),
        "intra_op_threads": args.intra_op_threads,
        "torch_threads": torch.get_num_threads(),
    }
    out = results.new_results(
        profile=args.profile,
        stage="export",
        candidate=f"Python onnxruntime {ort.__version__} {session.get_providers()[0]} on onnx/model.onnx; reference hidden "
        f"states from sentence-transformers modules on torch {torch.__version__} fp32 CPU, same ids",
        producer={
            "name": "model-pipeline/reference/encoder/stage_e5_export.py",
            "repository_commit": fixtures.git_head(Path(__file__).parent),
        },
        environment=env,
        assets=[
            {"path": "onnx/model.onnx", "sha256": fixtures.sha256_file(model_path)},
            {"path": "model.safetensors", "sha256": fixtures.sha256_file(args.model_dir / "model.safetensors")},
        ],
    )

    ort_hidden, torch_hidden = [], []
    for case in cases:
        ids = np.array([case["input_ids"]], dtype=np.int64)
        mask = np.ones_like(ids)
        hidden = ort_run(session, ids, mask)
        features = {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}
        features["token_type_ids"] = torch.zeros_like(features["input_ids"])
        with torch.inference_mode():
            reference = model(features)["token_embeddings"].numpy()
        ort_hidden.append((case["id"], hidden.shape[1:], hidden[0].ravel().tolist()))
        torch_hidden.append((case["id"], reference.shape[1:], reference[0].ravel().tolist()))
        out["cases"][case["id"]] = {
            "input_ids": case["input_ids"],
            "embedding": fixtures.float32_list(pooled(hidden, mask)[0]),
        }
    out["tensors"]["last_hidden_state"] = results.write_tensor(args.out, "export-last-hidden-state.f32", ort_hidden)
    out["tensors"]["reference_last_hidden_state"] = results.write_tensor(
        args.out, "export-reference-last-hidden-state.f32", torch_hidden
    )

    # The protected padded batch, as the tokenizer pads it, and every case in padded batches of 16.
    rows = batch_fx["rows"]
    ids = np.array([r["ids"] for r in rows], dtype=np.int64)
    mask = np.array([r["attention_mask"] for r in rows], dtype=np.int64)
    out["batches"]["fixture-padded"] = {
        "output": "embedding",
        "case_ids": [r["id"] for r in rows],
        "values": [fixtures.float32_list(v) for v in pooled(ort_run(session, ids, mask), mask)],
    }
    values = []
    for start in range(0, len(cases), BATCH_SIZE):
        chunk = cases[start : start + BATCH_SIZE]
        ids, mask = padded([c["input_ids"] for c in chunk])
        values += [fixtures.float32_list(v) for v in pooled(ort_run(session, ids, mask), mask)]
    out["batches"][f"all-cases-batch{BATCH_SIZE}"] = {
        "output": "embedding",
        "case_ids": [c["id"] for c in cases],
        "values": values,
    }
    fixtures.write_json(args.out / f"export{results.RESULTS_SUFFIX}", out)
    print(f"wrote {args.out / ('export' + results.RESULTS_SUFFIX)}: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
