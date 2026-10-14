"""Stage 1 and stage 2 results for GLiNER2 entity extraction (backlog P-015, evaluator task entity-extraction).

Writes stage results (`model_pipeline.results`, format version 2, task `entity-extraction`) for
`model-pipeline evaluate`. Applies no thresholds of its own: the case threshold comes from the
protected fixtures, and every bound from the profile's protected tolerances file.

    tokenizer  Re-runs GLiNER2's own Python processor (`ExtractorCollator`, inference mode, fast
               tokenizer) on each fixture case and records normalized text, words, character maps,
               input ids, attention mask, word and schema positions. This checks that the fixture
               processor path reproduces in this environment; it is not the owned (C#) processor,
               whose stage 1 results the C# fidelity tool writes in the same format.
    export     Python ONNX Runtime on the exported route A graph, fed the fixture inputs: count
               logits, span logits over valid spans (label, start word, width; row-major, invalid
               spans dropped, the layout of `span_logits_valid`), entities decoded from those logits
               with `gliner2_entity.decode_entities`, the ORT hidden state, and the PyTorch hidden
               state recomputed on the same ids (sidecar tensors). Padded batches group cases by label
               count (the graph's input contract) in batches of up to 8.

Usage (the P-014 export environment, requirements-export.txt, plus `pip install --no-deps -e model-pipeline`):
    python stage_gliner2.py tokenizer --model-dir <snapshot> --fixtures <dir with extraction.jsonl> \
        --profile gliner2-base-v1/onnx-fp32-cpu --out <results dir>
    python stage_gliner2.py export --model-dir <snapshot> --graph <gliner2-A-dynamo.onnx> \
        --fixtures <dir> --profile gliner2-base-v1/onnx-fp32-cpu --out <results dir> [--threads N]

CUDA profile (export only; environment requirements-cuda.txt): add `--provider cuda` and pass
`--profile gliner2-base-v1/onnx-fp32-cuda`. ONNX Runtime then runs on the CUDA execution provider with use_tf32=0;
the PyTorch reference hidden states stay fp32 CPU.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np

from model_pipeline import fixtures, results

TASK = "entity-extraction"
PACKAGES = ["gliner2", "torch", "transformers", "tokenizers", "onnxruntime", "numpy"]


def load_model(model_dir: Path):
    import torch
    from gliner2 import GLiNER2

    torch.manual_seed(0)
    model = GLiNER2.from_pretrained(str(model_dir))
    model.eval()
    return model


def schema_of(row: dict):
    from gliner2.inference.schema import Schema

    if row.get("descriptions"):
        return Schema().entities({label: row["descriptions"].get(label, {}) for label in row["labels_requested"]})
    return Schema().entities(row["labels_requested"])


def producer() -> dict:
    return {
        "name": "model-pipeline/reference/gliner2/stage_gliner2.py",
        "repository_commit": fixtures.git_head(Path(__file__).parent),
    }


def stage_tokenizer(args) -> int:
    from gliner2.training.trainer import ExtractorCollator

    model = load_model(args.model_dir)
    rows = fixtures.read_jsonl(args.fixtures / "extraction.jsonl")
    env = fixtures.environment(PACKAGES) | {"machine": platform.node(), "provider": "cpu"}
    out = results.new_results(
        profile=args.profile,
        stage="tokenizer",
        candidate="GLiNER2 Python processor (ExtractorCollator, inference mode, fast tokenizer): reproducibility of "
        "the fixture processor path, not the owned C# processor",
        producer=producer(),
        environment=env,
        assets=[{"path": "tokenizer.json", "sha256": fixtures.sha256_file(args.model_dir / "tokenizer.json")}],
        task=TASK,
    )
    for row in rows:
        collator = ExtractorCollator(
            model.processor, is_training=False, max_len=row.get("max_len"), architecture="span"
        )
        batch = collator([(row["text"], schema_of(row))])
        words = batch.text_word_indices[0, : batch.text_word_counts[0]]
        out["cases"][row["id"]] = {
            "normalized_text": batch.original_texts[0],
            "words": list(batch.text_tokens[0]),
            "start_map": [int(v) for v in batch.start_mappings[0]],
            "end_map": [int(v) for v in batch.end_mappings[0]],
            "input_ids": batch.input_ids[0].tolist(),
            "attention_mask": batch.attention_mask[0].tolist(),
            "word_index": words.tolist(),
            "schema_index": [int(v) for v in batch.schema_special_indices[0][0]],
        }
    path = args.out / f"tokenizer{results.RESULTS_SUFFIX}"
    fixtures.write_json(path, out)
    print(f"wrote {path}: {len(rows)} cases")
    return 0


def valid_logits(logits: np.ndarray, n: int, m: int, k: int) -> list[float]:
    """[M, W, K] padded logits to the fixture layout: valid spans only, row-major over (label, start, width)."""
    valid = (np.arange(n)[:, None] + np.arange(k)[None, :]) < n
    return fixtures.float32_list(np.stack([logits[q, :n, :k][valid] for q in range(m)]).reshape(-1))


def entity_result(row: dict, count: np.ndarray, logits: np.ndarray) -> dict:
    from gliner2_entity import decode_entities

    decoded = decode_entities(
        count_logits=count,
        span_logits=logits,
        labels=row["labels"],
        text=row["normalized_text"],
        start_map=row["start_map"],
        end_map=row["end_map"],
        threshold=row["threshold"],
    )
    items = [
        {"label": label, "text": e["text"], "start": e["start"], "end": e["end"], "confidence": e["confidence"]}
        for label, entries in decoded["entities"].items()
        for e in entries
    ]
    return {"gate_open": bool(int(np.argmax(count)) > 0), "entities": items}


def stage_export(args) -> int:
    import onnxruntime as ort
    import torch

    model = load_model(args.model_dir)
    rows = fixtures.read_jsonl(args.fixtures / "extraction.jsonl")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider_options: dict = {}
    if args.provider == "cuda":
        # The CUDA profiles pin use_tf32=0 (protected tolerances); there is no silent CPU fallback.
        provider_options = {"use_tf32": 0}
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.graph), options, providers=providers)
    if args.provider == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise SystemExit(f"--provider cuda: the session has no CUDA execution provider ({session.get_providers()})")
    data_file = args.graph.with_name(args.graph.name + ".data")
    assets = [{"path": f"{args.graph_asset_prefix}.onnx", "sha256": fixtures.sha256_file(args.graph)}]
    if data_file.exists():
        assets.append({"path": f"{args.graph_asset_prefix}.onnx.data", "sha256": fixtures.sha256_file(data_file)})
    assets.append({"path": "model.safetensors", "sha256": fixtures.sha256_file(args.model_dir / "model.safetensors")})
    env = fixtures.environment(PACKAGES) | {
        "machine": platform.node(),
        "provider": args.provider,
        "provider_options": provider_options,
        "onnxruntime_providers": session.get_providers(),
        "intra_op_threads": args.threads,
        "torch_threads": torch.get_num_threads(),
    }
    out = results.new_results(
        profile=args.profile,
        stage="export",
        candidate=f"Python onnxruntime {ort.__version__} {session.get_providers()[0]} on {args.graph.name} (route A, fixture "
        f"inputs); entities decoded with gliner2_entity.decode_entities; reference hidden states from the GLiNER2 "
        f"encoder on torch {torch.__version__} fp32 CPU, same ids",
        producer=producer(),
        environment=env,
        assets=assets,
        task=TASK,
    )

    def run(ids, mask, wi, si):
        count, logits, hidden = session.run(
            None, {"input_ids": ids, "attention_mask": mask, "word_index": wi, "schema_index": si}
        )
        return count, logits, hidden

    ort_hidden, torch_hidden = [], []
    for row in rows:
        ids = np.array([row["input_ids"]], dtype=np.int64)
        mask = np.ones_like(ids)
        wi = np.array([row["word_index"]], dtype=np.int64)
        si = np.array([row["schema_index"]], dtype=np.int64)
        count, logits, hidden = run(ids, mask, wi, si)
        with torch.inference_mode():
            ref = model.encoder(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask))
        ref_hidden = ref.last_hidden_state.numpy()
        n, m, k = len(row["start_map"]), len(row["labels"]), row["span_logits_shape"][2]
        out["cases"][row["id"]] = {
            "input_ids": row["input_ids"],
            "word_index": row["word_index"],
            "schema_index": row["schema_index"],
            "count_logits": fixtures.float32_list(count[0]),
            "span_logits": valid_logits(logits[0], n, m, k),
            "entities": entity_result(row, count[0], logits[0]),
        }
        ort_hidden.append((row["id"], hidden.shape[1:], hidden[0].ravel().tolist()))
        torch_hidden.append((row["id"], ref_hidden.shape[1:], ref_hidden[0].ravel().tolist()))
        print(f"{row['id']:30s} S={ids.shape[1]:5d} entities={len(out['cases'][row['id']]['entities']['entities'])}")
    out["tensors"]["last_hidden_state"] = results.write_tensor(args.out, "export-last-hidden-state.f32", ort_hidden)
    out["tensors"]["reference_last_hidden_state"] = results.write_tensor(
        args.out, "export-reference-last-hidden-state.f32", torch_hidden
    )

    # Padded batches: cases with the same label count (route A needs one label count per batch), corpus order.
    groups: dict[int, list[dict]] = {}
    for row in rows:
        if len(row["input_ids"]) <= args.batch_max_tokens:
            groups.setdefault(len(row["labels"]), []).append(row)
    case_ids, span_rows, count_rows = [], [], []
    for _m, members in sorted(groups.items()):
        for start in range(0, len(members), args.batch_size):
            chunk = members[start : start + args.batch_size]
            if len(chunk) < 2:
                continue
            s = max(len(r["input_ids"]) for r in chunk)
            w = max(len(r["word_index"]) for r in chunk)
            ids = np.zeros((len(chunk), s), np.int64)
            mask = np.zeros((len(chunk), s), np.int64)
            wi = np.zeros((len(chunk), w), np.int64)
            si = np.array([r["schema_index"] for r in chunk], dtype=np.int64)
            for j, r in enumerate(chunk):
                ids[j, : len(r["input_ids"])] = r["input_ids"]
                mask[j, : len(r["input_ids"])] = 1
                wi[j, : len(r["word_index"])] = r["word_index"]
            count, logits, _hidden = run(ids, mask, wi, si)
            for j, r in enumerate(chunk):
                n, m, k = len(r["start_map"]), len(r["labels"]), r["span_logits_shape"][2]
                case_ids.append(r["id"])
                span_rows.append(valid_logits(logits[j], n, m, k))
                count_rows.append(fixtures.float32_list(count[j]))
    name = f"by-label-count-batch{args.batch_size}"
    out["batches"][name] = {"output": "span_logits", "case_ids": case_ids, "values": span_rows}
    out["batches"][f"{name}-count"] = {"output": "count_logits", "case_ids": case_ids, "values": count_rows}
    path = args.out / f"export{results.RESULTS_SUFFIX}"
    fixtures.write_json(path, out)
    print(f"wrote {path}: {len(rows)} cases, {len(case_ids)} batched rows")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("tokenizer", "export"):
        p = sub.add_parser(name)
        p.add_argument("--model-dir", type=Path, required=True)
        p.add_argument("--fixtures", type=Path, required=True, help="directory with extraction.jsonl")
        p.add_argument("--profile", required=True)
        p.add_argument("--out", type=Path, required=True)
    exp = sub.choices["export"]
    exp.add_argument("--graph", type=Path, required=True, help="route A graph (external data next to it)")
    exp.add_argument(
        "--graph-asset-prefix",
        default="onnx/gliner2-entity-route-a",
        help="path the graph is declared under in the results' assets; must match the manifest's origin.files",
    )
    exp.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0: ORT default)")
    exp.add_argument("--batch-size", type=int, default=8)
    exp.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    exp.add_argument("--batch-max-tokens", type=int, default=100000, help="leave longer cases out of the batches")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return stage_tokenizer(args) if args.stage == "tokenizer" else stage_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
