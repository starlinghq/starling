"""Release step for the EmbeddingGemma 2 text-path graph (package Starlings.Models.EmbeddingGemma2). CPU only.

Turns the dynamo export of `export_embeddinggemma2_onnx.py` (`eg2-text-dynamo-sdpa.onnx` plus its external data
file) into the one file the C# package locks and Starling hosts on Hugging Face
(`Starlinghq/embeddinggemma-2-onnx`, release `embeddinggemma-2-onnx-r1`, admission decision in
`model-pipeline/admissions/embeddinggemma-2.md`):

    1. load the graph with its external data,
    2. clear every `metadata_props` entry (model, graph, values, initializers, nodes, functions): the dynamo exporter
       writes source stack traces there, which differ between runs, checkouts and machines,
    3. clear `doc_string` fields for the same reason,
    4. save it as a single file with the weights inline (about 1.09 GB, below protobuf's 2 GiB limit, so no external
       data file is needed),
    5. run `onnx.checker` and print the file's size and SHA-256 as a JSON record (`--expect-sha256` fails on a
       different digest).

The graph maps `input_ids` and `attention_mask` (int64, [batch, sequence]) to `last_hidden_state` ([batch, sequence,
512], after the final norm; diagnostic) and `token_embeddings` ([batch, sequence, 768], after `embedding_projection`;
what the package pools). Mean pooling, L2 normalization, prompts and Matryoshka truncation stay outside the graph.

The pinned recipe (checkpoint google/embeddinggemma-2 at 914f7f89142e33e77833254d9c9b90c3cef7303b, transformers 5.19.0,
torch 2.14.1, onnx 1.23.1, onnxscript 0.7.2, dynamo exporter, opset 20, environment `requirements.txt`):

    python export_embeddinggemma2_onnx.py --model-dir <snapshot 914f7f89> --out <dir> --exporters dynamo
    python release_embeddinggemma2_onnx.py --export <dir>/eg2-text-dynamo-sdpa.onnx \\
        --out <release>/onnx/embeddinggemma2-text-fp32.onnx

Two independent exports of the same snapshot give the same digest (checked when the asset was made; see
`ai-dotnet/src/Starlings.Models.EmbeddingGemma2/EmbeddingGemma2ModelAssets.cs`). The digest goes into the
asset lock and the registry manifests. `NOTICE-embeddinggemma2-onnx.txt` next to this script is the notice attached to
the release beside the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx


def _strip_nodes(nodes) -> int:
    cleared = 0
    for node in nodes:
        if len(node.metadata_props) or node.doc_string:
            cleared += 1
        del node.metadata_props[:]
        node.doc_string = ""
        for attr in node.attribute:
            if attr.HasField("g"):
                cleared += _strip_graph(attr.g)
            for g in attr.graphs:
                cleared += _strip_graph(g)
    return cleared


def _strip_graph(graph) -> int:
    del graph.metadata_props[:]
    graph.doc_string = ""
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        value.doc_string = ""
        del value.metadata_props[:]
    for init in graph.initializer:
        init.doc_string = ""
        del init.metadata_props[:]
    return _strip_nodes(graph.node)


def normalize(model: onnx.ModelProto) -> int:
    """Clears run-dependent text in place; returns the number of nodes that carried some."""
    del model.metadata_props[:]
    model.doc_string = ""
    cleared = _strip_graph(model.graph)
    for function in model.functions:
        del function.metadata_props[:]
        function.doc_string = ""
        cleared += _strip_nodes(function.node)
    return cleared


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export", type=Path, required=True, help="eg2-text-dynamo-sdpa.onnx from export_embeddinggemma2_onnx.py"
    )
    parser.add_argument("--out", type=Path, required=True, help="single-file graph to write")
    parser.add_argument("--expect-sha256", help="fail unless the written file has this digest")
    args = parser.parse_args()

    model = onnx.load(str(args.export), load_external_data=True)
    cleared = normalize(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.out), save_as_external_data=False)
    onnx.checker.check_model(str(args.out))
    digest = sha256(args.out)
    record = {
        "source": args.export.name,
        "path": args.out.name,
        "size": args.out.stat().st_size,
        "sha256": digest,
        "nodes": len(model.graph.node),
        "nodes_with_cleared_text": cleared,
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "producer": f"{model.producer_name} {model.producer_version}",
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "onnx": onnx.__version__,
    }
    print(json.dumps(record, indent=2))
    if args.expect_sha256 and digest != args.expect_sha256:
        print(f"digest {digest} differs from the expected {args.expect_sha256}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
