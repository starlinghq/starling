"""Browser spike (docs/m0/browser-wasm-spike.md): reduced-precision variants of the e5-small ONNX graph.

Exploratory. The variants are separate candidate profiles (plan 10.1), not the reference: their drift is measured
against the protected fp32 PyTorch embeddings by `drift_e5_browser.py`, natively here and in the browser.

Variants, all from the pinned snapshot's `onnx/model.onnx` (intfloat/multilingual-e5-small@614241f6):

- `fp16`: every float initializer and op converted to float16 by `onnxruntime.transformers.float16`
  (`convert_float_to_float16`, `keep_io_types=True`: inputs stay int64, `last_hidden_state` stays float32 through a
  final Cast). No other graph change.
- `int8-dynamic`: `onnxruntime.quantization.quantize_dynamic` with signed 8-bit weights (`QuantType.QInt8`),
  per-tensor, default op set (MatMul and Gather, so the 250k x 384 word-embedding table is int8 too). Activations are
  quantized at run time (DynamicQuantizeLinear, uint8).
- `int8-dynamic-pc`: the same with per-channel weight scales (`per_channel=True`).
- `int8-embed`: weight-only, only the Gather inputs quantized (`op_types_to_quantize=["Gather"]`): the word, position
  and type embedding tables become int8 with a DequantizeLinear, every MatMul stays fp32. The word embeddings are
  96 M of the 118 M parameters, so this keeps most of the size reduction without run-time activation quantization.

Environment: `requirements-browser.txt` (onnxruntime 1.30.0, onnx 1.23.1). The output directory gets the graphs and
`variants.json` (SHA-256, sizes, node-type counts, the tool versions and the exact calls).

Usage:
    python quantize_e5_browser.py --model <snapshot>/onnx/model.onnx --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
from pathlib import Path

import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.float16 import convert_float_to_float16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path, call: str) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    ops = collections.Counter(node.op_type for node in model.graph.node)
    dtypes = collections.Counter(onnx.TensorProto.DataType.Name(init.data_type) for init in model.graph.initializer)
    return {
        "file": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "node_types": dict(sorted(ops.items())),
        "initializer_dtypes": dict(sorted(dtypes.items())),
        "call": call,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="the snapshot's onnx/model.onnx (fp32)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    variants = [describe(args.model, "source: intfloat/multilingual-e5-small@614241f6 onnx/model.onnx")]
    variants[0]["variant"] = "fp32"

    fp16_path = args.out / "model-fp16.onnx"
    fp16_call = "convert_float_to_float16(onnx.load(model), keep_io_types=True)"
    fp16 = convert_float_to_float16(onnx.load(str(args.model)), keep_io_types=True)
    onnx.save(fp16, str(fp16_path))
    variants.append({"variant": "fp16", **describe(fp16_path, fp16_call)})

    int8_path = args.out / "model-int8-dynamic.onnx"
    int8_call = "quantize_dynamic(model, out, weight_type=QuantType.QInt8, per_channel=False, reduce_range=False)"
    quantize_dynamic(
        str(args.model), str(int8_path), weight_type=QuantType.QInt8, per_channel=False, reduce_range=False
    )
    variants.append({"variant": "int8-dynamic", **describe(int8_path, int8_call)})

    pc_path = args.out / "model-int8-dynamic-pc.onnx"
    pc_call = "quantize_dynamic(model, out, weight_type=QuantType.QInt8, per_channel=True, reduce_range=False)"
    quantize_dynamic(str(args.model), str(pc_path), weight_type=QuantType.QInt8, per_channel=True, reduce_range=False)
    variants.append({"variant": "int8-dynamic-pc", **describe(pc_path, pc_call)})

    embed_path = args.out / "model-int8-embed.onnx"
    embed_call = "quantize_dynamic(model, out, weight_type=QuantType.QInt8, op_types_to_quantize=['Gather'])"
    quantize_dynamic(str(args.model), str(embed_path), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    variants.append({"variant": "int8-embed", **describe(embed_path, embed_call)})

    record = {
        "tools": {
            "onnxruntime": onnxruntime.__version__,
            "onnx": onnx.__version__,
            "python": platform.python_version(),
        },
        "variants": variants,
    }
    (args.out / "variants.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for v in variants:
        print(f"{v['variant']:>13}  {v['bytes'] / 1e6:8.1f} MB  {v['sha256']}")


if __name__ == "__main__":
    main()
