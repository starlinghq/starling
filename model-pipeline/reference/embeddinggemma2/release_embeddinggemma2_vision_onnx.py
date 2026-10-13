"""Release step for the EmbeddingGemma 2 image graphs (release `embeddinggemma-2-vision-onnx-r1`). CPU only.

Produces the two files the C# package locks for image embeddings (`EmbeddingGemma2ModelAssets.ImageLock`) and that
Starling hosts on Hugging Face (`Starlinghq/embeddinggemma-2-vision-onnx`) next to
`NOTICE-embeddinggemma2-vision-onnx.txt`:

1. `embeddinggemma2-vision-fp32.onnx`: `eg2_vision.VisionPath` (patch embedder, 16 encoder layers, 3x3 pooling,
   sqrt(768) scale, `embed_vision`) exported with the dynamo exporter (opset 20; dynamic batch and patches,
   `9 * slots` with slots 1 to 1,120), then normalized as the text release step does
   (`release_embeddinggemma2_onnx.normalize`:
   run-dependent metadata and doc strings removed) and saved as one file with the weights inline. Inputs
   `pixel_values` (float32, [batch, patches, 768]) and `pixel_position_ids` (int64, [batch, patches, 2]); outputs
   `soft_tokens` ([batch, patches / 9, 512]), `pooled` ([batch, patches / 9, 768], diagnostic) and `soft_token_mask`
   (bool, [batch, patches / 9]).
2. `embeddinggemma2-text-mm-fp32.onnx`: the released text graph (`embeddinggemma2-text-fp32.onnx`, release
   `embeddinggemma-2-onnx-r1`, checked by SHA-256) rewritten by `eg2_vision.add_multimodal_inputs_release`: inputs
   `multimodal_embeds` (float32, [batch, sequence, 512]) and `multimodal_mask` (bool, [batch, sequence]) put soft tokens
   in place of the scaled token embedding, and a second projection of the soft tokens is mixed into the per-layer
   embedding projection. The released nodes keep their inputs, so ONNX Runtime optimizes them as in the released
   graph: a text-only call (mask all false) equals the released graph bit for bit (checked here at
   `ORT_ENABLE_ALL`). No re-export: same weights, same nodes plus five.

Checks after writing: `onnx.checker`; ONNX Runtime CPU vision graph vs PyTorch on random patches (padded and
unpadded); multimodal graph text-only vs the released graph (identical arrays required, else exit 1). The JSON record
gives sizes and SHA-256; `--expect-vision-sha256` and `--expect-text-mm-sha256` fail on another digest.

The pinned recipe (checkpoint google/embeddinggemma-2 at 914f7f89142e33e77833254d9c9b90c3cef7303b, transformers 5.19.0,
torch 2.14.1, onnx 1.23.1, onnxscript 0.7.2, onnxruntime 1.30.0, environment `requirements.txt`):

    python release_embeddinggemma2_vision_onnx.py --model-dir <snapshot 914f7f89> \\
        --text-graph <release>/onnx/embeddinggemma2-text-fp32.onnx --work <scratch dir> --out <release>/onnx

Two independent runs of the same snapshot give byte-identical files (checked when the assets were made;
`docs/evidence/embeddinggemma2-images/`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from eg2_vision import VisionPath, add_multimodal_inputs_release, load_model
from export_embeddinggemma2_vision_onnx import check_vision, export_vision
from release_embeddinggemma2_onnx import normalize

RELEASED_TEXT_SHA256 = "02cc7ecccbfcb29d812eb104e4c6ea6c599bc87e94f01082606f6a11f0252d11"
VISION_NAME = "embeddinggemma2-vision-fp32.onnx"
TEXT_MM_NAME = "embeddinggemma2-text-mm-fp32.onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path, model: onnx.ModelProto, cleared: int) -> dict:
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": sha256(path),
        "nodes": len(model.graph.node),
        "nodes_with_cleared_text": cleared,
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "producer": f"{model.producer_name} {model.producer_version}",
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
    }


def text_only_identical(mm: Path, released: Path) -> dict:
    """The multimodal graph with an all-false mask against the released graph, ORT CPU at ORT_ENABLE_ALL."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    a = ort.InferenceSession(str(released), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(mm), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20261011)
    worst = 0.0
    identical = True
    for batch, seq in ((1, 14), (2, 300), (3, 700)):
        ids = rng.integers(3, 200000, (batch, seq)).astype(np.int64)
        ids[:, 0] = 2
        mask = np.ones_like(ids)
        if batch > 1:
            mask[-1, seq // 2 :] = 0
            ids[-1, seq // 2 :] = 0
        x = a.run(None, {"input_ids": ids, "attention_mask": mask})
        y = b.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": mask,
                "multimodal_embeds": np.zeros((batch, seq, 512), np.float32),
                "multimodal_mask": np.zeros((batch, seq), bool),
            },
        )
        for p, q in zip(x, y, strict=True):
            worst = max(worst, float(np.abs(p - q).max()))
            identical &= bool(np.array_equal(p, q))
    return {
        "max_abs": worst,
        "identical": identical,
        "optimization": "ORT_ENABLE_ALL",
        "shapes": [[1, 14], [2, 300], [3, 700]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="snapshot of google/embeddinggemma-2 at 914f7f89")
    parser.add_argument("--text-graph", type=Path, required=True, help="released embeddinggemma2-text-fp32.onnx")
    parser.add_argument("--work", type=Path, required=True, help="scratch directory for the raw export")
    parser.add_argument("--out", type=Path, required=True, help="directory for the two release files")
    parser.add_argument("--expect-vision-sha256")
    parser.add_argument("--expect-text-mm-sha256")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    record: dict = {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}

    released_sha = sha256(args.text_graph)
    if released_sha != RELEASED_TEXT_SHA256:
        print(f"{args.text_graph}: SHA-256 {released_sha} is not the released text graph {RELEASED_TEXT_SHA256}")
        return 1

    torch.manual_seed(0)
    vision = VisionPath(load_model(args.model_dir)).eval()
    t0 = time.time()
    raw = args.work / "eg2-vision-dynamo-raw.onnx"
    with torch.inference_mode(False), torch.no_grad():
        export_vision(vision, "dynamo", raw)
    model = onnx.load(str(raw), load_external_data=True)
    cleared = normalize(model)
    vision_path = args.out / VISION_NAME
    onnx.save_model(model, str(vision_path), save_as_external_data=False)
    onnx.checker.check_model(str(vision_path))
    entry = describe(vision_path, model, cleared)
    del model
    entry["seconds"] = round(time.time() - t0, 1)
    entry["ort_cpu_vs_torch_random_patches"] = check_vision(vision, vision_path)
    record["vision"] = entry
    del vision

    t0 = time.time()
    model = onnx.load(str(args.text_graph), load_external_data=True)
    added = add_multimodal_inputs_release(model)
    cleared = normalize(model)
    mm_path = args.out / TEXT_MM_NAME
    onnx.save_model(model, str(mm_path), save_as_external_data=False)
    onnx.checker.check_model(str(mm_path))
    entry = describe(mm_path, model, cleared)
    del model
    entry["seconds"] = round(time.time() - t0, 1)
    entry["source"] = {"path": args.text_graph.name, "sha256": released_sha}
    entry["added_tensors"] = added
    entry["text_only_vs_released"] = text_only_identical(mm_path, args.text_graph)
    record["text_mm"] = entry

    print(json.dumps(record, indent=2))
    status = 0
    if not record["text_mm"]["text_only_vs_released"]["identical"]:
        print("the multimodal graph does not reproduce the released graph on text-only input")
        status = 1
    for key, expect in (("vision", args.expect_vision_sha256), ("text_mm", args.expect_text_mm_sha256)):
        if expect and record[key]["sha256"] != expect:
            print(f"{key}: digest {record[key]['sha256']} differs from the expected {expect}")
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
