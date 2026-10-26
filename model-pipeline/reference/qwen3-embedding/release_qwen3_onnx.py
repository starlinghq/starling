"""Release step for the Qwen3 Embedding series 0.6B graphs (hosted as `Starlinghq/qwen3-embedding-0.6b-onnx` and
`Starlinghq/qwen3-reranker-0.6b-onnx`, tag `r1`). CPU only.

Turns an export of `export_qwen3_onnx.py` (`model.onnx` plus its external data) into the files Starling hosts:

    1. load the graph with its external data,
    2. clear every `metadata_props` entry (model, graph, values, initializers, nodes, functions): the dynamo exporter
       writes source stack traces there, which differ between runs, checkouts and machines,
    3. clear `doc_string` fields for the same reason,
    4. save it as `model.onnx` with every tensor of at least 1 KiB in one external data file `model.onnx.data`
       (about 2.4 GB, above protobuf's 2 GiB limit for a single file); an existing data file is removed first, because
       onnx appends to it,
    5. run `onnx.checker` on the saved graph,
    6. copy the upstream files the hosted repository carries, unchanged (`tokenizer.json`, `config.json`,
       `tokenizer_config.json`, and `config_sentence_transformers.json` for the embedding model or
       `chat_template.jinja` and `1_LogitScore/config.json` for the reranker),
    7. write `build-lock.json` (upstream repository, revision and file digests, tool versions, the export record of
       `export.json`, and the size and SHA-256 of every released file) and print the release record.

`--expect-sha256` and `--expect-data-sha256` fail on a different digest of `model.onnx` or `model.onnx.data`.

The pinned recipe (environment `requirements.txt` next to this script):

    python export_qwen3_onnx.py --model embedding --model-dir <Qwen/Qwen3-Embedding-0.6B@97b0c614> --out <export>
    python release_qwen3_onnx.py --model embedding --model-dir <same snapshot> --export <export>/model.onnx \\
        --out <release>
    python export_qwen3_onnx.py --model reranker --model-dir <Qwen/Qwen3-Reranker-0.6B@e61197ed> --out <export>
    python release_qwen3_onnx.py --model reranker --model-dir <same snapshot> --export <export>/model.onnx \\
        --out <release>

Two independent exports of each snapshot give the same bytes. `NOTICE-qwen3-embedding-onnx.txt` and
`NOTICE-qwen3-reranker-onnx.txt` next to this script are the notices hosted beside the graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import onnx

SOURCES = {
    "embedding": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "copied": ["tokenizer.json", "config.json", "tokenizer_config.json", "config_sentence_transformers.json"],
    },
    "reranker": {
        "repo": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "copied": [
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "1_LogitScore/config.json",
        ],
    },
}
WEIGHTS = "model.safetensors"
DATA = "model.onnx.data"
SIZE_THRESHOLD = 1024


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


def file_record(path: Path, name: str) -> dict:
    return {"path": name, "size": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(SOURCES))
    parser.add_argument("--model-dir", type=Path, required=True, help="the pinned upstream snapshot")
    parser.add_argument("--export", type=Path, required=True, help="model.onnx from export_qwen3_onnx.py")
    parser.add_argument("--out", type=Path, required=True, help="release directory to write")
    parser.add_argument("--expect-sha256", help="fail unless the written model.onnx has this digest")
    parser.add_argument("--expect-data-sha256", help="fail unless the written model.onnx.data has this digest")
    args = parser.parse_args()
    source = SOURCES[args.model]

    model = onnx.load(str(args.export), load_external_data=True)
    cleared = normalize(model)
    args.out.mkdir(parents=True, exist_ok=True)
    graph, data = args.out / "model.onnx", args.out / DATA
    data.unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(graph),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=DATA,
        size_threshold=SIZE_THRESHOLD,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(graph))

    for name in source["copied"]:
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.model_dir / name, target)

    export_record = json.loads((args.export.parent / "export.json").read_text(encoding="utf-8"))
    released = [file_record(graph, "model.onnx"), file_record(data, DATA)]
    released += [file_record(args.out / name, name) for name in source["copied"]]
    record = {
        "schema": "starlings-onnx-build-lock/1",
        "model": args.model,
        "source": {
            "repo": source["repo"],
            "revision": source["revision"],
            "files": [file_record(args.model_dir / n, n) for n in [WEIGHTS, *source["copied"]]],
        },
        "recipe": {
            "repository": "https://github.com/starlinghq/starling",
            "directory": "model-pipeline/reference/qwen3-embedding",
            "environment": "requirements.txt",
            "commands": [
                f"python export_qwen3_onnx.py --model {args.model} --model-dir <snapshot> --out <export>",
                f"python release_qwen3_onnx.py --model {args.model} --model-dir <snapshot> "
                "--export <export>/model.onnx --out <release>",
            ],
        },
        "export": {k: v for k, v in export_record.items() if k != "attempts"}
        | {
            "attempts": [
                {
                    k: v
                    for k, v in a.items()
                    if k not in ("op_types", "traceback_tail", "graph_sha256", "graph_bytes", "seconds")
                }
                for a in export_record["attempts"]
            ]
        },
        "release": {
            "nodes": len(model.graph.node),
            "nodes_with_cleared_text": cleared,
            "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
            "producer": f"{model.producer_name} {model.producer_version}",
            "inputs": [i.name for i in model.graph.input],
            "outputs": [o.name for o in model.graph.output],
            "external_data_size_threshold": SIZE_THRESHOLD,
            "onnx": onnx.__version__,
            "python": sys.version.split()[0],
        },
        "files": released,
    }
    (args.out / "build-lock.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"files": released, "release": record["release"]}, indent=1))
    failed = False
    for expected, actual, name in (
        (args.expect_sha256, released[0]["sha256"], "model.onnx"),
        (args.expect_data_sha256, released[1]["sha256"], DATA),
    ):
        if expected and expected != actual:
            print(f"{name}: digest {actual} differs from the expected {expected}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
