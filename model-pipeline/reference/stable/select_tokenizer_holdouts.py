"""Select the proposed P-030 tokenizer (and text-encoder) holdouts, one disjoint corpus per tokenizer profile.

Sources (``sources.py``, pinned by commit, verified by SHA-256):

- MASSIVE 1.1 test split (CC-BY-4.0): short utterances in 52 locales and 20+ scripts.
- HumanEval prompts (MIT): Python code with indentation, docstrings and doctest output.
- OpenAssistant oasst1 validation split (Apache-2.0): assistant replies with Markdown, lists, code, numbers and
  long multi-paragraph text (the truncation path above 512 tokens).

Procedure (deterministic for a seed): each stratum is sorted by its source id, filtered, shuffled once with
``random.Random(f"{seed}:{stratum}")``, and dealt round-robin to the four profiles, so the four holdouts are
disjoint and none of them is chosen per profile by hand.

- MASSIVE: per locale, ``--per-locale`` utterances per profile; text as given (``utt``).
- HumanEval: ``--code`` prompts per profile; text is the ``prompt`` field.
- oasst1: ``--chat`` assistant messages per profile with ``review_result`` true, not deleted, not synthetic,
  no ``pii`` label above 0, detoxify toxicity below 0.1, 200 to 6,000 characters.
- Every text that also appears in a development corpus (``reference/corpus/*.jsonl``,
  ``reference/generation/prompts*.jsonl``) is excluded. Empty and duplicate texts are excluded.

Output per profile: ``<out>/<family>/tokenizer-holdout/corpus.jsonl`` in the corpus format of
``reference/corpus/`` (``id``, ``category``, ``text``), plus ``holdout.json`` with provenance, counts and six
suggested padded-batch ids (the runners' ``BATCH_IDS``).

Usage:
    python -I select_tokenizer_holdouts.py --cache <dir> --out model-pipeline/proposed/stable [--seed 20261014]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
import sources  # noqa: E402

PROFILES = {
    # profile -> (family directory, tokenizer description)
    "xlmr-unigram-e5": (
        "multilingual-e5-small",
        "XLM-R SentencePiece unigram, intfloat/multilingual-e5-small@614241f6",
    ),
    "gemma-bpe-embeddinggemma2": ("embeddinggemma-2", "Gemma BPE, google/embeddinggemma-2@914f7f89"),
    "qwen35-bpe-chat": ("qwen3.5-0.8b", "Qwen byte-level BPE (chat), Qwen/Qwen3.5-0.8B@2fc06364"),
    "deberta-v3-unigram-gliner2": (
        "gliner2-base-v1",
        "DeBERTa-v3 SentencePiece unigram, fastino/gliner2-base-v1@f9634218",
    ),
}
MASSIVE = "all_1.1/test/0000.parquet"
HUMANEVAL = "openai_humaneval/test-00000-of-00001.parquet"
OASST = "data/validation-00000-of-00001-134b8fd0c89408b6.parquet"


def _dev_texts(repo_root: Path) -> set[str]:
    texts: set[str] = set()
    ref = repo_root / "model-pipeline" / "reference"
    for path in sorted((ref / "corpus").glob("*.jsonl")) + sorted((ref / "generation").glob("prompts*.jsonl")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" in row:
                texts.add(row["text"])
            for m in row.get("messages", []):
                texts.add(m.get("content", ""))
    return texts


def _deal(items: list[dict], stratum: str, seed: int, per_profile: int) -> dict[str, list[dict]]:
    items = list(items)
    random.Random(f"{seed}:{stratum}").shuffle(items)
    names = list(PROFILES)
    need = per_profile * len(names)
    if len(items) < need:
        raise ValueError(f"{stratum}: {len(items)} eligible items, need {need}")
    dealt: dict[str, list[dict]] = {n: [] for n in names}
    for i, item in enumerate(items[:need]):
        dealt[names[i % len(names)]].append(item)
    return dealt


def _oasst_ok(row: dict) -> bool:
    labels = row.get("labels") or {}
    names, values = labels.get("name") or [], labels.get("value") or []
    pii = max((v for n, v in zip(names, values, strict=False) if n == "pii"), default=0.0)
    tox = (row.get("detoxify") or {}).get("toxicity")
    return (
        row["role"] == "assistant"
        and row.get("review_result") is True
        and row.get("deleted") is False
        and row.get("synthetic") is False
        and pii == 0
        and tox is not None
        and tox < 0.1
        and 200 <= len(row["text"]) <= 6000
    )


def select(cache: Path, seed: int, per_locale: int, code: int, chat: int, dev: set[str]) -> tuple[dict, dict]:
    out: dict[str, list[dict]] = {n: [] for n in PROFILES}
    counts: dict[str, dict[str, int]] = {n: {} for n in PROFILES}
    seen: set[str] = set(dev)

    def keep(text: str) -> bool:
        if not text or not text.strip() or text in seen:
            return False
        seen.add(text)
        return True

    massive = pq.read_table(sources.fetch("massive", MASSIVE, cache), columns=["id", "locale", "utt"]).to_pylist()
    by_locale: dict[str, list[dict]] = {}
    for row in sorted(massive, key=lambda r: (r["locale"], int(r["id"]))):
        by_locale.setdefault(row["locale"], []).append(row)
    for locale, rows in sorted(by_locale.items()):
        rows = [r for r in rows if keep(r["utt"])]
        for name, picked in _deal(rows, f"massive:{locale}", seed, per_locale).items():
            for r in picked:
                out[name].append(
                    {"id": f"massive-{locale}-{r['id']}", "category": f"massive:{locale}", "text": r["utt"]}
                )
            counts[name]["massive"] = counts[name].get("massive", 0) + len(picked)

    he = pq.read_table(sources.fetch("humaneval", HUMANEVAL, cache), columns=["task_id", "prompt"]).to_pylist()
    he = [r for r in sorted(he, key=lambda r: int(r["task_id"].split("/")[1])) if keep(r["prompt"])]
    for name, picked in _deal(he, "humaneval", seed, code).items():
        for r in picked:
            num = r["task_id"].split("/")[1]
            out[name].append({"id": f"humaneval-{num}", "category": "code:python", "text": r["prompt"]})
        counts[name]["humaneval"] = len(picked)

    oa = pq.read_table(sources.fetch("oasst1", OASST, cache)).to_pylist()
    oa = [r for r in sorted(oa, key=lambda r: r["message_id"]) if _oasst_ok(r) and keep(r["text"])]
    for name, picked in _deal(oa, "oasst1", seed, chat).items():
        for r in picked:
            out[name].append({"id": f"oasst1-{r['message_id']}", "category": f"chat:{r['lang']}", "text": r["text"]})
        counts[name]["oasst1"] = len(picked)
    return out, counts


def batch_ids(rows: list[dict]) -> list[str]:
    """Six rows of very different lengths and scripts for the runners' padded-batch check (deterministic)."""

    def first(pred) -> str:
        return next(r["id"] for r in rows if pred(r))

    by_len = sorted(rows, key=lambda r: (len(r["text"]), r["id"]))
    picks = [
        by_len[0]["id"],
        first(lambda r: r["category"] == "massive:ja-JP"),
        first(lambda r: r["category"] == "massive:ar-SA"),
        first(lambda r: r["category"] == "massive:th-TH"),
        first(lambda r: r["category"] == "code:python"),
        by_len[-1]["id"],
    ]
    return list(dict.fromkeys(picks))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--per-locale", type=int, default=4)
    ap.add_argument("--code", type=int, default=8)
    ap.add_argument("--chat", type=int, default=16)
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    rows, counts = select(args.cache, args.seed, args.per_locale, args.code, args.chat, _dev_texts(repo_root))
    for name, (family, description) in PROFILES.items():
        target = args.out / family / "tokenizer-holdout"
        sources.write_jsonl(target / "corpus.jsonl", rows[name])
        sources.write_json(
            target / "holdout.json",
            {
                "profile": name,
                "tokenizer": description,
                "status": "PROPOSED: candidate P-030 holdout, not accepted",
                "selection": {
                    "script": "model-pipeline/reference/stable/select_tokenizer_holdouts.py",
                    "seed": args.seed,
                    "per_locale": args.per_locale,
                    "code": args.code,
                    "chat": args.chat,
                    "counts": counts[name],
                    "rows": len(rows[name]),
                },
                "batch_ids": batch_ids(rows[name]),
                "sources": [
                    sources.provenance("massive", [MASSIVE]),
                    sources.provenance("humaneval", [HUMANEVAL]),
                    sources.provenance("oasst1", [OASST]),
                ],
            },
        )
        print(f"{name}: {len(rows[name])} rows -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
