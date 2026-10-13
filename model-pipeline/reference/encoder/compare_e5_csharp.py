"""Compare C# ONNX Runtime outputs with Python ONNX Runtime outputs on the same provider and settings.

Inputs: a C# dump (`<name>.bin` plus `<name>.json`, written by the throwaway probe described in
docs/m0/e5-gpu.md) and a Python dump (`<name>.npz` from `run_e5_cuda.py --dump`). Both hold, per
case of the protected `embeddings.jsonl` in file order, the last hidden state and the pooled embedding.

Usage:
    python compare_e5_csharp.py --fixtures <protected onnx-fp32-cpu dir> --csharp <dir> --python <dir> \
        --pair <csharp-name>=<python-name> [--pair ...] --out <json>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model_pipeline import fixtures

DIM = 384


def summary(a: np.ndarray) -> dict:
    a = np.abs(np.asarray(a, dtype=np.float64)).ravel()
    return {"max_abs": float(a.max()), "mean_abs": float(a.mean()), "p99_abs": float(np.quantile(a, 0.99))}


def read_csharp(path: Path, cases: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = np.fromfile(path, dtype="<f4")
    out, pos = {}, 0
    for case in cases:
        n = len(case["input_ids"]) * DIM
        out[case["id"]] = (data[pos : pos + n].reshape(-1, DIM), data[pos + n : pos + n + DIM])
        pos += n + DIM
    assert pos == data.size, (pos, data.size)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--csharp", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = fixtures.read_jsonl(args.fixtures / "embeddings.jsonl")
    result = {}
    for pair in args.pair:
        cs_name, py_name = pair.split("=", 1)
        cs = read_csharp(args.csharp / f"{cs_name}.bin", cases)
        py = np.load(args.python / f"{py_name}.npz")
        hid = [cs[c["id"]][0] - py[f"hidden/{c['id']}"] for c in cases]
        emb = [cs[c["id"]][1] - py[f"embedding/{c['id']}"] for c in cases]
        identical = sum(int(np.array_equal(cs[c["id"]][0], py[f"hidden/{c['id']}"])) for c in cases)
        result[f"{cs_name} vs {py_name}"] = {
            "last_hidden_state": summary(np.concatenate([h.ravel() for h in hid])),
            "hidden_bit_identical_cases": identical,
            "cases": len(cases),
            "embedding_csharp_pooling_vs_numpy_pooling": summary(np.concatenate(emb)),
        }
        r = result[f"{cs_name} vs {py_name}"]
        emb_max = r["embedding_csharp_pooling_vs_numpy_pooling"]["max_abs"]
        print(
            f"{cs_name} vs {py_name}: hidden max {r['last_hidden_state']['max_abs']:.3g} "
            f"identical {identical}/{len(cases)} emb max {emb_max:.3g}"
        )
    fixtures.write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
