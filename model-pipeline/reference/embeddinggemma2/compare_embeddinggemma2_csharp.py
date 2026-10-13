"""Compare the throwaway C# smoke's embeddings with Python ONNX Runtime dumps (admission spike).

The C# smoke (outside the repository, see docs/m0/embeddinggemma2-spike.md) feeds the fixture ids from
embeddings.jsonl to the exported graph, mean-pools with double accumulation and L2-normalizes, and
writes row-major float32 files. The Python dump comes from parity_embeddinggemma2_ort.py --dump.

Usage:
    python compare_embeddinggemma2_csharp.py --csharp <prefix> --python <npz> --reference <dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model_pipeline import fixtures


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    cos = (a64 * b64).sum(1) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "min_cosine": float(cos.min())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csharp", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = fixtures.read_jsonl(args.reference / "embeddings.jsonl")
    ref = np.stack([np.array(r["embedding"], dtype=np.float32) for r in rows])
    cs = np.fromfile(args.csharp + ".f32", dtype=np.float32).reshape(len(rows), -1)
    cs_b = np.fromfile(args.csharp + ".batched.f32", dtype=np.float32).reshape(len(rows), -1)
    py = np.load(args.python)
    result = {
        "rows": len(rows),
        "csharp_vs_python_ort_single": stats(cs, py["embeddings"]),
        "csharp_vs_python_ort_batched": stats(cs_b, py["batched"]),
        "csharp_vs_reference": stats(cs, ref),
        "python_ort_vs_reference": stats(py["embeddings"], ref),
    }
    fixtures.write_json(args.out, result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
