"""Release step for the GLiNER2 route A graph (backlog P-015). CPU, no torch needed.

Turns the route A dynamo export of `export_gliner2_onnx.py` (`gliner2-A-dynamo.onnx` plus its external data file)
into the one file the C# package locks and Starling hosts on Hugging Face (`Starlinghq/gliner2-base-v1-onnx`, with
`NOTICE-gliner2-onnx.txt`):

    1. load the graph with its external data,
    2. clear every `metadata_props` entry (model, graph, nodes, functions): the dynamo exporter writes source stack
       traces there, which differ between runs and machines; nothing else in the file does (P-014, section 5),
    3. clear `doc_string` fields for the same reason,
    4. save it as a single file with the weights inline (830 MB, below protobuf's 2 GB limit),
    5. run `onnx.checker` and print the file's size and SHA-256 as a JSON record.

The pinned recipe (P-014 route A, dynamo exporter, torch 2.14.1, opset 18, `requirements-export.txt`):

    python export_gliner2_onnx.py --model-dir <snapshot f9634218> --out <dir> --routes A --exporters dynamo
    python release_gliner2_onnx.py --export <dir>/gliner2-A-dynamo.onnx --out <release>/onnx/gliner2-entities-fp32.onnx

Two exports of the same snapshot give the same digest (checked on two P-014 exports). The digest goes into the
asset lock (`ai-dotnet/src/Starlings.Models.Gliner2/Gliner2ModelAssets.cs`) and the manifest.
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
            if attr.g is not None and attr.g.node:
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
    parser.add_argument("--export", type=Path, required=True, help="gliner2-A-dynamo.onnx from export_gliner2_onnx.py")
    parser.add_argument("--out", type=Path, required=True, help="single-file graph to write")
    args = parser.parse_args()

    model = onnx.load(str(args.export), load_external_data=True)
    cleared = normalize(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.out), save_as_external_data=False)
    onnx.checker.check_model(str(args.out))
    record = {
        "source": args.export.name,
        "path": args.out.name,
        "size": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "nodes": len(model.graph.node),
        "nodes_with_cleared_text": cleared,
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "producer": f"{model.producer_name} {model.producer_version}",
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "onnx": onnx.__version__,
    }
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
