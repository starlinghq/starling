"""Python ONNX Runtime outputs of one GLiNER2 graph on the reference inputs, for the C# comparison (P-014).

Writes one JSON line per case: count logits and span logits over valid spans (same layout as
`span_logits_valid` in `extraction.jsonl`). The C# smoke reads this file and reports its own
distance to it, so a C# difference is separated from an export difference (plan section 8).

Usage:
    python dump_gliner2_ort.py --onnx-dir <dir> --graph A-dynamo --reference <ref dir> --out <jsonl> \
        [--provider cpu|cuda] [--threads N] [--cuda-tf32 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from parity_gliner2_ort import Graph, load_reference

from model_pipeline import fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--graph", default="A-dynamo")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cuda-tf32", type=int, default=0)
    args = parser.parse_args()

    graph = Graph(args.graph, args.onnx_dir, args.provider, args.threads, args.cuda_tf32)
    rows = []
    for row, ref in load_reference(args.reference):
        _hidden, count, logits = graph.run(
            ref["input_ids"], ref["attention_mask"], ref["word_index"], ref["schema_index"]
        )
        n = len(row["start_map"])
        k = row["span_logits_shape"][2]
        valid = (np.arange(n)[:, None] + np.arange(k)[None, :]) < n
        lg = np.stack([logits[0, q, :n][valid] for q in range(len(row["labels"]))])
        rows.append(
            {
                "id": row["id"],
                "graph": args.graph,
                "provider": args.provider,
                "count_logits": [float(str(np.float32(v))) for v in count[0]],
                "span_logits_valid": [float(str(np.float32(v))) for v in lg.reshape(-1)],
            }
        )
    fixtures.write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
