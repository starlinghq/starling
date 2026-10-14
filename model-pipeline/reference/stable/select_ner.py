"""Select the proposed G3 NER subsets for GLiNER2 (R5 gold format) from Few-NERD and WNUT-17.

Sources (``sources.py``): Few-NERD ``supervised`` test split (CC-BY-SA-4.0) and the WNUT-17 test set
``emerging.test.annotated`` (CC-BY-4.0), both pinned by commit and SHA-256.

Procedure (deterministic for a seed):

- Sentences are sorted by source order, filtered to 5 to 60 tokens, shuffled once with
  ``random.Random(f"{seed}:{name}")``, and the first ``--sentences`` kept, then re-sorted by source order.
  Sentences without entities are kept (they test empty results).
- Text is the tokens joined by single spaces (the sources are pre-tokenized); offsets are Unicode code points
  of that text, end exclusive.
- Few-NERD uses IO tags: a maximal run of one coarse type is one entity (the dataset's own convention). The
  coarse type ``other`` is dropped from gold and not queried. Labels queried: the dataset's coarse class names,
  verbatim, in class order.
- WNUT-17 uses BIO tags; labels are the six class names verbatim with ``-`` kept (``creative-work``).

Also writes ``extraction-release-v1.jsonl``: the first ``--release`` rows of each subset (source order) in the
extraction-corpus format of ``reference/corpus/gliner2-extraction-corpus-v1.jsonl`` (``id``, ``category``, ``text``,
``labels``, ``threshold`` 0.5), the input of ``reference/gliner2/run_gliner2_extraction.py`` for a held-out G1/G2
extraction release set (P-030).

Usage:
    python -I select_ner.py --cache <dir> --out model-pipeline/proposed/stable/gliner2-base-v1/ner [--seed 20261014]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
import sources  # noqa: E402

FEWNERD = "supervised/test-00000-of-00001.parquet"
FEWNERD_CLASSES = ["O", "art", "building", "event", "location", "organization", "other", "person", "product"]
FEWNERD_DROP = {"other"}
WNUT = "emerging.test.annotated"
WNUT_LABELS = ["person", "location", "corporation", "product", "creative-work", "group"]


def _text_and_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return " ".join(tokens), offsets


def _spans(tags: list[str], offsets: list[tuple[int, int]], bio: bool) -> list[dict]:
    spans, start, label = [], None, None
    for i, tag in enumerate([*tags, "O"]):
        kind, _, name = tag.partition("-") if bio else ("I" if tag != "O" else "O", "", tag)
        begins = tag != "O" and (kind == "B" or name != label or start is None)
        if start is not None and (tag == "O" or begins):
            spans.append({"start": offsets[start][0], "end": offsets[i - 1][1], "label": label})
            start, label = None, None
        if begins:
            start, label = i, name
    return spans


def _sample(rows: list[dict], name: str, seed: int, n: int) -> list[dict]:
    eligible = [r for r in rows if 5 <= len(r["tokens"]) <= 60]
    order = {id(r): i for i, r in enumerate(eligible)}
    picked = list(eligible)
    random.Random(f"{seed}:{name}").shuffle(picked)
    return sorted(picked[:n], key=lambda r: order[id(r)])


def fewnerd(cache: Path, seed: int, n: int) -> list[dict]:
    table = pq.read_table(sources.fetch("fewnerd", FEWNERD, cache), columns=["id", "tokens", "ner_tags"]).to_pylist()
    rows = sorted(table, key=lambda r: int(r["id"]))
    labels = [c for c in FEWNERD_CLASSES[1:] if c not in FEWNERD_DROP]
    out = []
    for r in _sample(rows, "fewnerd", seed, n):
        text, offsets = _text_and_offsets(r["tokens"])
        tags = [FEWNERD_CLASSES[t] for t in r["ner_tags"]]
        ents = [e for e in _spans(tags, offsets, bio=False) if e["label"] not in FEWNERD_DROP]
        out.append({"id": f"fewnerd-test-{r['id']}", "text": text, "labels": labels, "entities": ents})
    return out


def wnut(cache: Path, seed: int, n: int) -> list[dict]:
    raw = sources.fetch("wnut17", WNUT, cache).read_text(encoding="utf-8")
    rows, tokens, tags = [], [], []
    for line in [*raw.split("\n"), ""]:
        if not line.strip():
            if tokens:
                rows.append({"index": len(rows), "tokens": tokens, "tags": tags})
            tokens, tags = [], []
            continue
        tok, tag = line.split("\t")
        tokens.append(tok)
        tags.append(tag)
    out = []
    for r in _sample(rows, "wnut17", seed, n):
        text, offsets = _text_and_offsets(r["tokens"])
        ents = _spans(r["tags"], offsets, bio=True)
        out.append({"id": f"wnut17-test-{r['index']}", "text": text, "labels": WNUT_LABELS, "entities": ents})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--sentences", type=int, default=300)
    ap.add_argument("--release", type=int, default=30)
    args = ap.parse_args()
    release = []
    for name, rows, key, path in (
        ("fewnerd-test-v1", fewnerd(args.cache, args.seed, args.sentences), "fewnerd", FEWNERD),
        ("wnut17-test-v1", wnut(args.cache, args.seed, args.sentences), "wnut17", WNUT),
    ):
        target = args.out / name
        sources.write_jsonl(target / "gold.jsonl", rows)
        n_ents = sum(len(r["entities"]) for r in rows)
        sources.write_json(
            target / "dataset.json",
            {
                "name": name,
                "status": "PROPOSED: candidate G3 NER set, not accepted; not gate evidence",
                "provenance": sources.provenance(key, [path]),
                "license": sources.SOURCES[key].license,
                "selection": {
                    "script": "model-pipeline/reference/stable/select_ner.py",
                    "seed": args.seed,
                    "sentences": len(rows),
                    "entities": n_ents,
                    "filters": "5 to 60 tokens; tokens joined by single spaces"
                    + ("; coarse type 'other' dropped" if key == "fewnerd" else ""),
                },
            },
        )
        print(f"{name}: {len(rows)} sentences, {n_ents} entities")
        release += [
            {"id": r["id"], "category": key, "text": r["text"], "labels": r["labels"], "threshold": 0.5}
            for r in rows[: args.release]
        ]
    sources.write_jsonl(args.out / "extraction-release-v1.jsonl", release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
