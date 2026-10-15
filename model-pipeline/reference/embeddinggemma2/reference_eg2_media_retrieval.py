"""Upstream reference score of EmbeddingGemma 2 on a text-to-image or text-to-video retrieval set (G3).

The set is a G3 retrieval dataset with a media corpus (`select_eg2_retrieval.py build`): text queries, and documents
that are image or video files pinned by SHA-256. This runner embeds every query and document with upstream
sentence-transformers (the same environment as `run_embeddinggemma2_image.py` / `run_embeddinggemma2_video.py`):
queries with the model's `query` prompt, images and videos without a prompt (model card: prefixes apply to text
only), one input per forward call. Images are passed as PIL images, videos as their path (torchcodec on the pinned
FFmpeg 8 LGPL build, as the video runner does). It then scores the embeddings with the evaluator's own functions
(`model_pipeline.evaluate_task_quality`), so the numbers are the ones `model-pipeline evaluate --g3 retrieval`
prints for the same embeddings.

Writes `--embeddings` (the evaluator's candidate format, outside the repository) and `--scores` (JSON: metrics, the
dataset digests, the embeddings' SHA-256, the rank of each query's relevant document, the margin between it and the
best other document, and the environment).

Usage (videos: LD_LIBRARY_PATH=<FFmpeg 8 LGPL lib dir>, EG2_TORCHCODEC_FFMPEG_MAJOR=8):
    python reference_eg2_media_retrieval.py --model-dir <snapshot> --revision <sha> --dataset <dir> --media <dir>
        --embeddings <file> --scores <file> [--device cpu|cuda] [--threads <n>]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

PROMPTS = {"query": "task: search result | query: "}
PACKAGES = ["torch", "torchvision", "torchcodec", "transformers", "sentence-transformers", "numpy", "pillow"]
ASSETS = ["config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json", "model.safetensors"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()

    from model_pipeline import evaluate_task_quality as tq

    data = tq.load_retrieval_set(args.dataset)
    if data.modality not in tq.MEDIA_MODALITIES:
        raise SystemExit(f"{args.dataset}: not a media corpus (modality {data.modality})")
    corpus = [json.loads(line) for line in (args.dataset / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in corpus:
        if sha256_file(args.media / row["file"]) != row["sha256"]:
            raise SystemExit(f"{row['file']}: SHA-256 differs from corpus.jsonl")

    if data.modality == "video":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import eg2_video

        eg2_video.pin_torchcodec_ffmpeg()

    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modality import infer_modality

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    st = SentenceTransformer(str(args.model_dir), device=args.device, model_kwargs={"dtype": torch.float32})
    st.eval()

    def embed(item, prompt: str | None) -> list[float]:
        kwargs = {"prompt": prompt} if prompt else {}
        features = st.preprocess([item], **kwargs)
        features = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in features.items()}
        with torch.inference_mode():
            out = st(features)
        return out["sentence_embedding"][0].float().cpu().numpy().tolist()

    rows = []
    t0 = time.time()
    for qid, text in data.queries.items():
        rows.append({"kind": "query", "id": qid, "prompt": "query", "embedding": embed(text, PROMPTS["query"])})
    t_queries = time.time() - t0
    for i, row in enumerate(corpus):
        path = args.media / row["file"]
        if data.modality == "image":
            with Image.open(path) as im:
                im.load()
                vec = embed(im, None)
        else:
            # As run_embeddinggemma2_video.py: a path string where sentence-transformers detects a video, else a dict.
            vec = embed(str(path) if infer_modality(str(path)) == "video" else {"video": str(path)}, None)
        rows.append({"kind": "document", "id": row["id"], "prompt": "none", "embedding": vec})
        print(f"{i + 1}/{len(corpus)} {row['id']}", flush=True)
    t_docs = time.time() - t0 - t_queries
    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.embeddings.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    emb = tq.load_embeddings(args.embeddings, data)
    ranking = tq.rank(emb)
    computed = tq.retrieval_metrics(ranking, data.qrels)

    def cos(a: list[float], b: list[float]) -> float:
        a, b = tq._normalize(a), tq._normalize(b)
        return sum(x * y for x, y in zip(a, b, strict=True))

    per_query = {}
    for q, docs in ranking.items():
        relevant = {d for d, r in data.qrels[q].items() if r > 0}
        rel = next(d for d in docs if d in relevant)
        other = next(d for d in docs if d not in relevant)
        per_query[q] = {
            "relevant_rank": docs.index(rel) + 1,
            "margin_vs_best_other": cos(emb.queries[q], emb.documents[rel]) - cos(emb.queries[q], emb.documents[other]),
        }
    margins = sorted(abs(v["margin_vs_best_other"]) for v in per_query.values())
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    report = {
        "dataset": {
            "path": str(args.dataset),
            "modality": data.modality,
            "sha256": tq.dataset_digest(data.files[:3]),
            "sha256_with_metadata": tq.dataset_digest(data.files) if len(data.files) > 3 else None,
            "queries": len(data.queries),
            "documents": len(data.corpus),
        },
        "reference": {
            "model": "google/embeddinggemma-2",
            "revision": args.revision,
            "assets": {a: sha256_file(args.model_dir / a) for a in ASSETS if (args.model_dir / a).is_file()},
            "prompts": {"query": PROMPTS["query"], "document": None},
            "dtype": "float32",
            "device": args.device,
            "tf32": False,
            "threads": torch.get_num_threads(),
        },
        "embeddings": {"path": str(args.embeddings), "sha256": sha256_file(args.embeddings)},
        "metrics": computed["metrics"],
        "smallest_abs_margins": margins[:10],
        "per_query": {q: computed["per_query"][q] | per_query[q] for q in sorted(per_query)},
        "seconds": {"queries": round(t_queries, 1), "documents": round(t_docs, 1)},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": versions,
            "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
    }
    if data.modality == "video":
        import torchcodec._core as core

        report["environment"]["ffmpeg"] = core.get_ffmpeg_library_versions()
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    m = computed["metrics"]
    print(f"ndcg@10 {m['ndcg@10']:.4f} mrr@10 {m['mrr@10']:.4f} recall@1 {m['recall@1']:.4f}")
    print(f"smallest |margin| {margins[0]:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
