"""Reference retrieval scores for the proposed G3 set: upstream PyTorch (sentence-transformers) or Python ORT.

Embeds every query and passage of an R5 retrieval dataset and writes the candidate format the evaluator reads
(``{"kind", "id", "prompt", "embedding"}`` per line), then scores it with the evaluator's own functions
(``model_pipeline.evaluate_task_quality``: exhaustive cosine ranking, nDCG@10, recall@k, MRR@10, and the top-10
overlap against ``--baseline``). The embeddings file stays outside Git; the scores file records its SHA-256.

Prompts are the ones the .NET producer applies (``Starlings.Fidelity.Tool retrieval``): e5 ``query: `` and
``passage: ``; EmbeddingGemma 2 its ``query`` and ``document`` prompts from ``config_sentence_transformers.json``.

Backends:
- ``st``: sentence-transformers on the snapshot (the upstream PyTorch reference), fp32, CPU.
- ``ort``: e5 only. Python ONNX Runtime CPU on ``--onnx`` (default ``onnx/model.onnx``; an int8 browser variant
  from ``encoder/quantize_e5_browser.py`` for the overlap metric), ``tokenizers`` with truncation to 512, mean
  pooling and L2 normalization as ``encoder/run_e5.py``.

Usage:
    python run: reference_retrieval.py --model e5|embeddinggemma2 --model-dir <snapshot> --dataset <dir> \\
        --embeddings <out.jsonl> --scores <out.json> [--backend st|ort] [--onnx <graph>] [--baseline <jsonl>]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from model_pipeline import evaluate_task_quality as tq
from model_pipeline import fixtures

PROMPTS = {
    "e5": {"query": "query: ", "document": "passage: "},
    "embeddinggemma2": {"query": "task: search result | query: ", "document": "title: none | text: "},
}


def _rows(dataset: Path, name: str) -> list[tuple[str, str]]:
    return [(r["id"], r["text"]) for r in fixtures.read_jsonl(dataset / name)]


def embed_st(model_dir: Path, texts: list[str], prompt: str, batch: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    global _ST
    if "_ST" not in globals():
        _ST = SentenceTransformer(str(model_dir), device="cpu")
    return _ST.encode(texts, prompt=prompt, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True)


def embed_ort(model_dir: Path, onnx_path: Path, texts: list[str], prompt: str, batch: int, threads: int) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(512)
    tok.enable_padding(pad_id=1, pad_token="<pad>", direction="right")
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), batch):
        encs = tok.encode_batch([prompt + t for t in texts[i : i + batch]])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(["last_hidden_state"], feeds)[0]
        m = mask[..., None].astype(np.float32)
        mean = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        out.append(mean / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None))
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--backend", choices=["st", "ort"], default="st")
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--baseline", type=Path, help="embeddings JSONL of the fp32 reference, for the top-10 overlap")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    rows = []
    for kind, name in (("query", "queries.jsonl"), ("document", "corpus.jsonl")):
        items = _rows(args.dataset, name)
        prompt = PROMPTS[args.model][kind]
        texts = [t for _, t in items]
        if args.backend == "st":
            vecs = embed_st(args.model_dir, texts, prompt, args.batch)
        else:
            if args.model != "e5":
                raise SystemExit("--backend ort is implemented for e5 only")
            vecs = embed_ort(
                args.model_dir,
                args.onnx or args.model_dir / "onnx" / "model.onnx",
                texts,
                prompt,
                args.batch,
                args.threads,
            )
        for (name_id, _), v in zip(items, vecs, strict=True):
            rows.append({"kind": kind, "id": name_id, "prompt": prompt, "embedding": fixtures.float32_list(v)})
    fixtures.write_jsonl(args.embeddings, rows)
    elapsed = time.perf_counter() - t0

    data = tq.load_retrieval_set(args.dataset)
    ranking = tq.rank(tq.load_embeddings(args.embeddings, data))
    result = tq.retrieval_metrics(ranking, data.qrels)
    scores = {
        "model": args.model,
        "backend": args.backend,
        "onnx": str(args.onnx) if args.onnx else None,
        "dataset_sha256": tq.dataset_digest(data.files),
        "embeddings_sha256": fixtures.sha256_file(args.embeddings),
        "metrics": result["metrics"],
        "per_query": result["per_query"],
        "embed_seconds": round(elapsed, 1),
        "environment": fixtures.environment(
            ["torch", "transformers", "sentence-transformers", "tokenizers", "onnxruntime", "numpy"]
        ),
    }
    if args.baseline:
        base = tq.rank(tq.load_embeddings(args.baseline, data))
        ov = tq.top10_overlap(ranking, base)
        scores["top10_overlap"] = {k: v for k, v in ov.items() if k != "per_query"}
        scores["top10_overlap"]["baseline_sha256"] = fixtures.sha256_file(args.baseline)
    fixtures.write_json(args.scores, scores)
    print(json.dumps({k: scores[k] for k in ("metrics", "embed_seconds")} | {"overlap": scores.get("top10_overlap")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
