"""Stage 2 (export) results for EmbeddingGemma 2: Python ONNX Runtime on the release graph against PyTorch.

Runs the release graph (`onnx/embeddinggemma2-text-fp32.onnx`, release_embeddinggemma2_onnx.py) in Python ONNX Runtime
on the protected fixture ids (every row of embeddings.jsonl: 46 texts x no prompt, query, document; prompt rows are
cases "<id>@<prompt>"), pools the token embeddings in NumPy float32 (mean over the mask, L2 normalization), and
recomputes the PyTorch token embeddings and hidden states with `eg2_text.TextPath` on the same ids. Writes
`export.results.json` in the stage results format plus float32 sidecars: `token_embeddings` and
`reference_token_embeddings` (768 wide, valid positions; bound float.token_embeddings_relative) and
`last_hidden_state` / `reference_last_hidden_state` (512 wide; report only in this profile). Applies no thresholds.

Usage (environment requirements.txt; CUDA: requirements-cuda.txt and --provider cuda, use_tf32=0):
    python stage_embeddinggemma2_export.py --model-dir <snapshot 914f7f89> --graph <release .onnx> \\
        --fixtures <protected onnx-fp32-cpu dir> --profile embeddinggemma-2/onnx-fp32-cpu --out <results dir>
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from eg2_text import TextPath, load_text_model, pool

from model_pipeline import fixtures, results

PAD_ID = 0
BATCH_SIZE = 8
PACKAGES = ["torch", "transformers", "onnxruntime", "onnx", "numpy"]


def case_id(row: dict) -> str:
    return row["id"] if row.get("prompt", "none") == "none" else f"{row['id']}@{row['prompt']}"


def padded(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(r) for r in rows)
    ids = np.array([r + [PAD_ID] * (width - len(r)) for r in rows], dtype=np.int64)
    mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=np.int64)
    return ids, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--intra-op-threads", type=int, default=0, help="0 keeps the ONNX Runtime default")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    rows = fixtures.read_jsonl(args.fixtures / "embeddings.jsonl")
    batch_fx = json.loads((args.fixtures / "tokenizer-batch.json").read_text(encoding="utf-8"))
    text_path = TextPath(load_text_model(str(args.model_dir))).eval()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    provider_options: dict = {}
    if args.provider == "cuda":
        provider_options = {"use_tf32": 0}
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.graph), options, providers=providers)

    def run(ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden, tokens = session.run(
            ["last_hidden_state", "token_embeddings"], {"input_ids": ids, "attention_mask": mask}
        )
        return hidden, tokens

    env = fixtures.environment(PACKAGES)
    env |= {
        "machine": platform.node(),
        "provider": args.provider,
        "provider_options": provider_options,
        "onnxruntime_providers": session.get_providers(),
        "intra_op_threads": args.intra_op_threads,
        "torch_threads": torch.get_num_threads(),
    }
    graph_sha = fixtures.sha256_file(args.graph)
    out = results.new_results(
        profile=args.profile,
        stage="export",
        candidate=f"Python onnxruntime {ort.__version__} {session.get_providers()[0]} on the release graph; reference "
        f"token embeddings from eg2_text.TextPath on torch {torch.__version__} fp32 CPU, same ids",
        producer={
            "name": "model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py",
            "repository_commit": fixtures.git_head(Path(__file__).parent),
        },
        environment=env,
        assets=[
            {"path": "onnx/embeddinggemma2-text-fp32.onnx", "sha256": graph_sha},
            {"path": "model.safetensors", "sha256": fixtures.sha256_file(args.model_dir / "model.safetensors")},
        ],
    )

    tokens_out, tokens_ref, hidden_out, hidden_ref = [], [], [], []
    for row in rows:
        cid = case_id(row)
        ids = np.array([row["input_ids"]], dtype=np.int64)
        mask = np.ones_like(ids)
        hidden, tokens = run(ids, mask)
        with torch.inference_mode():
            ref_h, ref_t = (t.numpy() for t in text_path(torch.from_numpy(ids), torch.from_numpy(mask)))
        tokens_out.append((cid, tokens.shape[1:], tokens[0].ravel().tolist()))
        tokens_ref.append((cid, ref_t.shape[1:], ref_t[0].ravel().tolist()))
        hidden_out.append((cid, hidden.shape[1:], hidden[0].ravel().tolist()))
        hidden_ref.append((cid, ref_h.shape[1:], ref_h[0].ravel().tolist()))
        out["cases"][cid] = {"input_ids": row["input_ids"], "embedding": fixtures.float32_list(pool(tokens, mask)[0])}
    out["tensors"]["token_embeddings"] = results.write_tensor(args.out, "export-token-embeddings.f32", tokens_out)
    out["tensors"]["reference_token_embeddings"] = results.write_tensor(
        args.out, "export-reference-token-embeddings.f32", tokens_ref
    )
    out["tensors"]["last_hidden_state"] = results.write_tensor(args.out, "export-last-hidden-state.f32", hidden_out)
    out["tensors"]["reference_last_hidden_state"] = results.write_tensor(
        args.out, "export-reference-last-hidden-state.f32", hidden_ref
    )

    # The protected padded batch (pad id 0) and every row in padded batches of 8.
    brows = batch_fx["rows"]
    ids = np.array([r["ids"] for r in brows], dtype=np.int64)
    mask = np.array([r["attention_mask"] for r in brows], dtype=np.int64)
    out["batches"]["fixture-padded"] = {
        "output": "embedding",
        "case_ids": [r["id"] for r in brows],
        "values": [fixtures.float32_list(v) for v in pool(run(ids, mask)[1], mask)],
    }
    values = []
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start : start + BATCH_SIZE]
        ids, mask = padded([r["input_ids"] for r in chunk])
        values += [fixtures.float32_list(v) for v in pool(run(ids, mask)[1], mask)]
    out["batches"][f"all-cases-batch{BATCH_SIZE}"] = {
        "output": "embedding",
        "case_ids": [case_id(r) for r in rows],
        "values": values,
    }
    fixtures.write_json(args.out / f"export{results.RESULTS_SUFFIX}", out)
    print(f"wrote {args.out / ('export' + results.RESULTS_SUFFIX)}: {len(rows)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
