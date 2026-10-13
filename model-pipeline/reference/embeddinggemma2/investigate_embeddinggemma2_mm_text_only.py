"""Why the spike's multimodal text graph differs from the released text graph on text-only input (about 1e-3).

Compares, on text-only input (all-false `multimodal_mask`), the released text graph against one or more multimodal
variants at every ONNX Runtime graph-optimization level, and lists the fused nodes whose attributes differ after
optimization. Inputs: the 8 text queries of the image fixtures plus random ids.

Finding (2026-10-11, ORT 1.30.0): the spike variant (`eg2_vision.add_multimodal_inputs`, one `Where` between
`Mul(embedding, embed_scale)` and its consumers) equals the released graph at `ORT_DISABLE_ALL` and `ORT_ENABLE_BASIC`
and differs from `ORT_ENABLE_EXTENDED` up (max abs 3.0e-3 on CPU, 4.6e-3 on CUDA, on token embeddings of about 2,000).
The only attribute difference after optimization is one `FusedMatMul` (`MatMulScaleFusion`) on the per-layer
embedding (PLE) projection: in the released graph ONNX Runtime folds `embed_scale` (sqrt(512)) and the projection
scale (1/sqrt(512)) into `alpha = 0.99999994` on the unscaled embedding; behind the `Where` it can fold only the
second (`alpha = 0.0442`) on the scaled embedding, which rounds differently. The release layout
(`add_multimodal_inputs_release`) leaves the released nodes' inputs untouched and is bit-identical at every level.

Usage:
    python investigate_embeddinggemma2_mm_text_only.py --released <embeddinggemma2-text-fp32.onnx>
        --variant spike=<graph> --variant release=<graph> --fixtures <protected image-fp32-cpu dir>
        [--provider cpu|cuda] [--opt-dir <scratch dir for optimized graphs>]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

LEVELS = {
    "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def session(path: Path, level: str, provider: str, optimized: Path | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = LEVELS[level]
    if optimized is not None:
        so.optimized_model_filepath = str(optimized)
        so.add_session_config_entry("session.optimized_model_external_initializers_file_name", optimized.name + ".data")
        so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    providers = (
        [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})] if provider == "cuda" else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(str(path), so, providers=providers)


def inputs(fixtures: Path) -> list[np.ndarray]:
    rows = [json.loads(x) for x in (fixtures / "image-embeddings.jsonl").read_text().splitlines() if x.strip()]
    out = [np.array([r["input_ids"]], np.int64) for r in rows if r["kind"] == "text"]
    rand = np.random.default_rng(7).integers(3, 200000, (2, 300)).astype(np.int64)
    rand[:, 0] = 2
    return [*out, rand]


def compare(released: Path, variant: Path, level: str, provider: str, feeds: list[np.ndarray]) -> dict:
    a = session(released, level, provider)
    b = session(variant, level, provider)
    worst, identical = 0.0, True
    for ids in feeds:
        mask = np.ones_like(ids)
        x = a.run(["token_embeddings"], {"input_ids": ids, "attention_mask": mask})[0]
        y = b.run(
            ["token_embeddings"],
            {
                "input_ids": ids,
                "attention_mask": mask,
                "multimodal_embeds": np.zeros((*ids.shape, 512), np.float32),
                "multimodal_mask": np.zeros(ids.shape, bool),
            },
        )[0]
        worst = max(worst, float(np.abs(x - y).max()))
        identical &= bool(np.array_equal(x, y))
    return {"max_abs": worst, "identical": identical}


def fused_signatures(path: Path) -> collections.Counter:
    model = onnx.load(str(path), load_external_data=False)
    sig = collections.Counter()
    for n in model.graph.node:
        attrs = []
        for a in n.attribute:
            v = onnx.helper.get_attribute_value(a)
            if isinstance(v, int | float):
                attrs.append(f"{a.name}={v}")
        sig[(n.op_type, n.domain, tuple(sorted(attrs)))] += 1
    return sig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--released", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="name=path of a multimodal graph")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opt-dir", type=Path, help="write ORT_ENABLE_EXTENDED graphs here and diff fused nodes")
    args = parser.parse_args()
    feeds = inputs(args.fixtures)
    report: dict = {"onnxruntime": ort.__version__, "provider": args.provider, "variants": {}}
    for spec in args.variant:
        name, path = spec.split("=", 1)
        report["variants"][name] = {lv: compare(args.released, Path(path), lv, args.provider, feeds) for lv in LEVELS}
        if args.opt_dir is not None:
            args.opt_dir.mkdir(parents=True, exist_ok=True)
            ref, var = args.opt_dir / "released-extended.onnx", args.opt_dir / f"{name}-extended.onnx"
            session(args.released, "extended", "cpu", ref)
            session(Path(path), "extended", "cpu", var)
            r, v = fused_signatures(ref), fused_signatures(var)
            report["variants"][name]["extended_node_differences"] = [
                {"node": list(k), "released": r[k], name: v[k]}
                for k in sorted(set(r) | set(v), key=str)
                if r[k] != v[k]
            ]
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
