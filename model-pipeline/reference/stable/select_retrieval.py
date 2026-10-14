"""Select the proposed G3 retrieval subset (e5, EmbeddingGemma 2, browser int8 overlap) from MIRACL dev.

Source: ``mteb/MIRACLRetrievalHardNegatives`` at a pinned commit (``sources.py``): MIRACL dev topics and
relevance judgments with a hard-negative corpus per language.

Procedure, per language in ``sources.MIRACL_LANGS`` (sorted, deterministic for a seed):

1. Queries sorted by id; keep those with at least one judgment of relevance 1 whose judged passages are all in
   the corpus. Draw ``--queries`` of them with ``random.Random(f"{seed}:{lang}:queries")``.
2. Corpus: every judged passage (relevance 0 or 1) of the drawn queries, plus ``--distractors`` passages drawn
   from the rest of the language's hard-negative corpus (sorted ids, ``random.Random(f"{seed}:{lang}:docs")``).
3. Passage text is ``title + " " + text`` when the title is non-empty (the MTEB/BEIR convention); query text
   as given. Ids are prefixed with the language (``en:0``, ``en:462221#4``) so languages share one dataset.

Writes ``queries.jsonl``, ``corpus.jsonl``, ``qrels.jsonl`` and ``dataset.json`` in the R5 retrieval format
(model-pipeline/README.md, "Task quality").

Usage:
    python -I select_retrieval.py --cache <dir> --out <dir> [--seed 20261014 --queries 25 --distractors 100]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
import sources  # noqa: E402


def _read(cache: Path, lang: str, part: str) -> list[dict]:
    path = f"{lang}-{part}/dev-00000-of-00001.parquet"
    return pq.read_table(sources.fetch("miracl-hn", path, cache)).to_pylist()


def select(cache: Path, seed: int, n_queries: int, n_distractors: int) -> tuple[list, list, list, dict]:
    queries, corpus, qrels, counts = [], [], [], {}
    for lang in sorted(sources.MIRACL_LANGS):
        docs = {r["id"]: r for r in _read(cache, lang, "corpus")}
        judged: dict[str, dict[str, int]] = {}
        for r in _read(cache, lang, "qrels"):
            judged.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
        texts = {str(r["id"]): r["text"] for r in _read(cache, lang, "queries")}
        eligible = sorted(
            (
                q
                for q, j in judged.items()
                if q in texts and any(v > 0 for v in j.values()) and all(d in docs for d in j)
            ),
            key=lambda q: (len(q), q),
        )
        drawn = sorted(random.Random(f"{seed}:{lang}:queries").sample(eligible, n_queries), key=lambda q: (len(q), q))
        doc_ids = sorted({d for q in drawn for d in judged[q]})
        rest = sorted(set(docs) - set(doc_ids))
        distractors = sorted(random.Random(f"{seed}:{lang}:docs").sample(rest, n_distractors))
        for q in drawn:
            queries.append({"id": f"{lang}:{q}", "text": texts[q]})
            for d, rel in sorted(judged[q].items()):
                qrels.append({"query_id": f"{lang}:{q}", "doc_id": f"{lang}:{d}", "relevance": rel})
        for d in doc_ids + distractors:
            title, body = (docs[d].get("title") or "").strip(), docs[d]["text"]
            corpus.append({"id": f"{lang}:{d}", "text": f"{title} {body}" if title else body})
        counts[lang] = {"queries": len(drawn), "judged_passages": len(doc_ids), "distractors": len(distractors)}
    return queries, corpus, qrels, counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--queries", type=int, default=25)
    ap.add_argument("--distractors", type=int, default=100)
    args = ap.parse_args()
    queries, corpus, qrels, counts = select(args.cache, args.seed, args.queries, args.distractors)
    sources.write_jsonl(args.out / "queries.jsonl", queries)
    sources.write_jsonl(args.out / "corpus.jsonl", corpus)
    sources.write_jsonl(args.out / "qrels.jsonl", qrels)
    paths = [
        f"{lang}-{p}/dev-00000-of-00001.parquet"
        for lang in sorted(sources.MIRACL_LANGS)
        for p in ("corpus", "queries", "qrels")
    ]
    sources.write_json(
        args.out / "dataset.json",
        {
            "name": "miracl-dev-hn-8lang-v1",
            "status": "PROPOSED: candidate G3 retrieval set, not accepted; not gate evidence until the maintainer "
            "commits it under model-pipeline/protected/",
            "provenance": sources.provenance("miracl-hn", paths),
            "license": "CC-BY-SA-4.0 (passages: Wikipedia, CC BY-SA; MIRACL topics and judgments: Apache-2.0)",
            "selection": {
                "script": "model-pipeline/reference/stable/select_retrieval.py",
                "seed": args.seed,
                "queries_per_language": args.queries,
                "distractors_per_language": args.distractors,
                "languages": sorted(sources.MIRACL_LANGS),
                "counts": counts,
            },
            "note": (
                "The Multilingual E5 technical report (arXiv:2402.05672) lists MIRACL training data in the e5 "
                "fine-tuning mix; this subset is the dev split. EmbeddingGemma 2 training data is not documented "
                "at this level."
            ),
        },
    )
    print(f"{len(queries)} queries, {len(corpus)} passages, {len(qrels)} judgments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
