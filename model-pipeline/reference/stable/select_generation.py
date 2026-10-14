"""Select the proposed Qwen3.5 sets: the P-030 generation release prompts and the G3 task-quality set.

Sources (``sources.py``): OpenAssistant oasst1 validation split (Apache-2.0) and GSM8K ``main`` test split
(MIT), pinned by commit and SHA-256.

Release prompts (P-030, G1/G2 on held-out prompts): oasst1 root prompts (``role`` prompter, no parent),
``review_result`` true, not deleted, not synthetic, no ``pii`` label above 0, detoxify toxicity below 0.1, 20 to
1,000 characters. Sorted by message id, shuffled once with ``random.Random(f"{seed}:oasst1-prompts")``, first
``--release`` kept. One user message each, the template's default thinking mode, ``max_new_tokens``
``--release-tokens``. Output format: ``reference/generation/prompts-v1.jsonl`` (input of ``run_reference.py``,
``make_generation_fixture.py`` and ``Starlings.Qwen35.Fidelity``).

Task set (G3): GSM8K test questions, sorted by row index, shuffled once with
``random.Random(f"{seed}:gsm8k")``, first ``--task`` kept. The user message is the question followed by the
instruction Qwen's model card uses for math ("Please reason step by step, and put your final answer within
\\boxed{}."); thinking disabled (``enable_thinking: false``) so the answer fits ``--task-tokens``. Each row carries
the reference answer (the number after ``####`` in the GSM8K solution, commas removed) in ``answer``.

Usage:
    python -I select_generation.py --cache <dir> --out model-pipeline/proposed/stable/qwen3.5-0.8b [--seed 20261014]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
import sources  # noqa: E402

OASST = "data/validation-00000-of-00001-134b8fd0c89408b6.parquet"
GSM8K = "main/test-00000-of-00001.parquet"
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def _prompt_ok(row: dict) -> bool:
    labels = row.get("labels") or {}
    names, values = labels.get("name") or [], labels.get("value") or []
    pii = max((v for n, v in zip(names, values, strict=False) if n == "pii"), default=0.0)
    tox = (row.get("detoxify") or {}).get("toxicity")
    return (
        row["role"] == "prompter"
        and row.get("parent_id") is None
        and row.get("review_result") is True
        and row.get("deleted") is False
        and row.get("synthetic") is False
        and pii == 0
        and tox is not None
        and tox < 0.1
        and 20 <= len(row["text"].strip()) <= 1000
    )


def release_prompts(cache: Path, seed: int, n: int, max_new: int) -> list[dict]:
    rows = pq.read_table(sources.fetch("oasst1", OASST, cache)).to_pylist()
    rows = [r for r in sorted(rows, key=lambda r: r["message_id"]) if _prompt_ok(r)]
    random.Random(f"{seed}:oasst1-prompts").shuffle(rows)
    return [
        {
            "id": f"oasst1-{r['message_id'][:8]}",
            "category": f"oasst1:{r['lang']}",
            "messages": [{"role": "user", "content": r["text"]}],
            "max_new_tokens": max_new,
        }
        for r in rows[:n]
    ]


def gsm8k_answer(solution: str) -> str:
    return solution.rsplit("####", 1)[1].strip().replace(",", "")


def task_set(cache: Path, seed: int, n: int, max_new: int) -> list[dict]:
    rows = pq.read_table(sources.fetch("gsm8k", GSM8K, cache)).to_pylist()
    indexed = list(enumerate(rows))
    random.Random(f"{seed}:gsm8k").shuffle(indexed)
    return [
        {
            "id": f"gsm8k-test-{i}",
            "category": "gsm8k",
            "messages": [{"role": "user", "content": f"{r['question']}\n{MATH_INSTRUCTION}"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_new_tokens": max_new,
            "answer": gsm8k_answer(r["answer"]),
        }
        for i, r in indexed[:n]
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--release", type=int, default=32)
    ap.add_argument("--release-tokens", type=int, default=64)
    ap.add_argument("--task", type=int, default=50)
    ap.add_argument("--task-tokens", type=int, default=512)
    args = ap.parse_args()
    rel = release_prompts(args.cache, args.seed, args.release, args.release_tokens)
    sources.write_jsonl(args.out / "release-prompts" / "prompts-release-v1.jsonl", rel)
    sources.write_json(
        args.out / "release-prompts" / "selection.json",
        {
            "status": "PROPOSED: candidate P-030 generation release prompts, not accepted",
            "script": "model-pipeline/reference/stable/select_generation.py",
            "seed": args.seed,
            "count": len(rel),
            "max_new_tokens": args.release_tokens,
            "provenance": sources.provenance("oasst1", [OASST]),
        },
    )
    task = task_set(args.cache, args.seed, args.task, args.task_tokens)
    target = args.out / "task-quality" / "gsm8k-test-v1"
    sources.write_jsonl(target / "prompts.jsonl", task)
    sources.write_json(
        target / "dataset.json",
        {
            "name": "gsm8k-test-v1",
            "status": "PROPOSED: candidate G3 generation task set, not accepted; not gate evidence",
            "provenance": sources.provenance("gsm8k", [GSM8K]),
            "license": "MIT",
            "metric": "exact_match: the last \\boxed{...} of the output (else the last number), commas and a "
            "trailing '.0' removed, equals `answer` (reference/stable/score_generation.py)",
            "selection": {
                "script": "model-pipeline/reference/stable/select_generation.py",
                "seed": args.seed,
                "count": len(task),
                "max_new_tokens": args.task_tokens,
                "instruction": MATH_INSTRUCTION,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    )
    print(f"release prompts: {len(rel)}; task rows: {len(task)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
