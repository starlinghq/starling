"""Export the EmbeddingGemma 2 image path to ONNX (image-path spike). CPU only.

Two graphs:

1. Vision graph (`eg2-vision-<exporter>.onnx`): `eg2_vision.VisionPath` (patch embedder, 16 encoder layers, 3x3
   pooling, sqrt(768) scale, `embed_vision`). Inputs `pixel_values` (float32, [batch, patches, 768]) and
   `pixel_position_ids` (int64, [batch, patches, 2]); outputs `soft_tokens` ([batch, patches / 9, 512]), `pooled`
   ([batch, patches / 9, 768], diagnostic) and `soft_token_mask` (bool, [batch, patches / 9]). `patches` is dynamic
   (2,520 upstream; any multiple of 9 that holds the image). Exporters: dynamo first, TorchScript as the fallback.
   The dynamo graph is then stripped (metadata, doc strings) and saved as one file, as the text release step does.
2. Multimodal text graph (`embeddinggemma2-text-mm-fp32.onnx`): the released text graph
   (`embeddinggemma2-text-fp32.onnx`, release `embeddinggemma-2-onnx-r1`) rewritten by
   `eg2_vision.add_multimodal_inputs`: two more inputs (`multimodal_embeds`, `multimodal_mask`) and one `Where` after
   the scaled token embedding. No re-export: the text path's nodes and weights are the released ones.

After each step the script checks the graph with `onnx.checker` and runs ONNX Runtime CPU once against PyTorch on
random patches (vision) or on random ids with a soft-token span (text); `export-vision.json` records the outcome,
node counts, sizes and digests.

Usage:
    python export_embeddinggemma2_vision_onnx.py --model-dir <snapshot> --text-graph <released text graph>
        --out <dir> [--exporters dynamo,legacy]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from eg2_vision import VisionPath, add_multimodal_inputs, load_model
from release_embeddinggemma2_onnx import _strip_graph

IMAGE_TOKEN = 258880


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_patches(counts: list[int], patches: int = 2520) -> tuple[torch.Tensor, torch.Tensor]:
    """Random pixels in [0, 1] for images of (grid_h, grid_w) patch grids, zero-padded to `patches` rows."""
    g = torch.Generator().manual_seed(sum(counts))
    pv = torch.zeros(len(counts) // 2, patches, 768)
    pos = torch.full((len(counts) // 2, patches, 2), -1, dtype=torch.long)
    for b in range(len(counts) // 2):
        gh, gw = counts[2 * b], counts[2 * b + 1]
        n = gh * gw
        pv[b, :n] = torch.rand(n, 768, generator=g)
        gx, gy = torch.meshgrid(torch.arange(gw), torch.arange(gh), indexing="xy")
        pos[b, :n] = torch.stack([gx, gy], -1).reshape(n, 2)
    return pv, pos


def export_vision(model: VisionPath, exporter: str, path: Path) -> None:
    pv, pos = sample_patches([39, 60, 12, 30])
    names = {
        "input_names": ["pixel_values", "pixel_position_ids"],
        "output_names": ["soft_tokens", "pooled", "soft_token_mask"],
    }
    if exporter == "dynamo":
        batch = torch.export.Dim("batch")
        patches = 9 * torch.export.Dim("slots", min=1, max=1120)  # pooling needs a multiple of 9
        torch.onnx.export(
            model,
            (pv, pos),
            str(path),
            dynamo=True,
            external_data=True,
            dynamic_shapes={"pixel_values": {0: batch, 1: patches}, "pixel_position_ids": {0: batch, 1: patches}},
            **names,
        )
    else:
        torch.onnx.export(
            model,
            (pv, pos),
            str(path),
            dynamo=False,
            opset_version=18,
            dynamic_axes={
                "pixel_values": {0: "batch", 1: "patches"},
                "pixel_position_ids": {0: "batch", 1: "patches"},
                "soft_tokens": {0: "batch", 1: "slots"},
                "pooled": {0: "batch", 1: "slots"},
                "soft_token_mask": {0: "batch", 1: "slots"},
            },
            **names,
        )


def strip_and_inline(src: Path, dst: Path) -> None:
    m = onnx.load(str(src), load_external_data=True)
    del m.metadata_props[:]
    m.doc_string = ""
    _strip_graph(m.graph)
    for f in m.functions:
        del f.metadata_props[:]
        f.doc_string = ""
    onnx.save(m, str(dst), save_as_external_data=False)


def node_counts(path: Path) -> dict:
    g = onnx.load(str(path), load_external_data=False)
    ops: dict[str, int] = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return {
        "opset": [{"domain": o.domain, "version": o.version} for o in g.opset_import],
        "nodes": len(g.graph.node),
        "op_types": dict(sorted(ops.items())),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def check_vision(model: VisionPath, path: Path) -> dict:
    onnx.checker.check_model(str(path), full_check=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    runs = {}
    for name, counts, patches in (("two-images-padded", [39, 60, 3, 150], 2520), ("one-image-unpadded", [12, 30], 360)):
        pv, pos = sample_patches(counts, patches)
        with torch.inference_mode():
            ref = [t.numpy() for t in model(pv, pos)]
        got = sess.run(None, {"pixel_values": pv.numpy(), "pixel_position_ids": pos.numpy()})
        valid = ref[2]
        runs[name] = {
            "soft_tokens_max_abs_valid": float(np.abs(got[0] - ref[0])[valid].max()),
            "pooled_max_abs_valid": float(np.abs(got[1] - ref[1])[valid].max()),
            "mask_equal": bool(np.array_equal(got[2], valid)),
            "finite": bool(all(np.isfinite(x).all() for x in got[:2])),
        }
    return runs


def check_text_mm(text_model, path: Path, released: Path) -> dict:
    """ORT on the rewritten graph vs PyTorch (inputs_embeds path), and vs the released graph on text only."""
    onnx.checker.check_model(str(path), full_check=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rel = ort.InferenceSession(str(released), providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(3, 200000, (2, 300), generator=g)
    ids[:, 0] = 2
    mask = torch.ones_like(ids)
    mask[1, 200:] = 0
    ids[1, 200:] = 0
    mm = torch.zeros(2, 300, dtype=torch.bool)
    mm[0, 10:270] = True
    mm[1, 5:50] = True
    ids[mm] = IMAGE_TOKEN
    soft = torch.randn(2, 300, 512, generator=g) * 0.05
    with torch.inference_mode():
        emb = text_model.embed_tokens(torch.where(mm, 0, ids))
        emb = torch.where(mm[..., None], soft, emb)
        ref = text_model(inputs_embeds=emb, attention_mask=mask).last_hidden_state.numpy()
    got = sess.run(
        None,
        {
            "input_ids": ids.numpy(),
            "attention_mask": mask.numpy(),
            "multimodal_embeds": soft.numpy(),
            "multimodal_mask": mm.numpy(),
        },
    )
    valid = mask.numpy().astype(bool)
    text_ids = torch.randint(3, 200000, (2, 64), generator=g)
    text_mask = torch.ones_like(text_ids)
    a = rel.run(None, {"input_ids": text_ids.numpy(), "attention_mask": text_mask.numpy()})
    b = sess.run(
        None,
        {
            "input_ids": text_ids.numpy(),
            "attention_mask": text_mask.numpy(),
            "multimodal_embeds": np.zeros((2, 64, 512), np.float32),
            "multimodal_mask": np.zeros((2, 64), bool),
        },
    )
    return {
        "token_embeddings_vs_torch_max_abs_valid": float(np.abs(got[1] - ref)[valid].max()),
        "token_embeddings_vs_torch_rel_max_valid": float(
            (np.abs(got[1] - ref) / np.maximum(1, np.abs(ref)))[valid].max()
        ),
        "text_only_vs_released_graph_max_abs": float(max(np.abs(x - y).max() for x, y in zip(a, b, strict=True))),
        "text_only_vs_released_graph_identical": bool(all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))),
        "finite": bool(np.isfinite(got[1][valid]).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exporters", default="dynamo,legacy")
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = load_model(args.model_dir)
    vision = VisionPath(model).eval()
    results = {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}

    for exporter in args.exporters.split(","):
        raw = args.out / f"eg2-vision-{exporter}-raw.onnx"
        t0 = time.time()
        try:
            with torch.inference_mode(False), torch.no_grad():
                export_vision(vision, exporter, raw)
            final = args.out / f"embeddinggemma2-vision-{exporter}-fp32.onnx"
            strip_and_inline(raw, final)
            entry = {"status": "exported", "seconds": round(time.time() - t0, 1)}
            entry.update(node_counts(final))
            entry["ort_cpu_vs_torch_random_patches"] = check_vision(vision, final)
        except Exception as exc:  # recorded as evidence; the stop rule is applied by the operator
            entry = {
                "status": "failed",
                "seconds": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}"[:4000],
                "traceback_tail": traceback.format_exc()[-4000:],
            }
        results[f"vision-{exporter}"] = entry
        print(exporter, json.dumps({k: v for k, v in entry.items() if k not in ("op_types", "traceback_tail")})[:2000])

    if not args.skip_text:
        t0 = time.time()
        graph = onnx.load(str(args.text_graph), load_external_data=True)
        add_multimodal_inputs(graph)
        dst = args.out / "embeddinggemma2-text-mm-fp32.onnx"
        onnx.save(graph, str(dst), save_as_external_data=False)
        del graph
        entry = {"status": "rewritten", "seconds": round(time.time() - t0, 1), "source_sha256": sha256(args.text_graph)}
        entry.update(node_counts(dst))
        entry["ort_cpu_checks"] = check_text_mm(model.language_model, dst, args.text_graph)
        results["text-mm"] = entry
        print("text-mm", json.dumps({k: v for k, v in entry.items() if k != "op_types"})[:2000])

    (args.out / "export-vision.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
