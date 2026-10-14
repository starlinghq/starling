"""Exact-match score of a Qwen3.5 run on the proposed GSM8K task set (G3, generation).

Reads the task set (``prompts.jsonl`` with ``answer`` per row) and a run's ``results.jsonl``: the transformers
reference (``reference/generation/run_reference.py``, field ``output_text``) or the .NET tool
(``Starlings.Qwen35.Fidelity``, field ``output_ids``, decoded here with ``--tokenizer``). Every task row must have
exactly one result.

Answer extraction (fixed before any run): the content of the last ``\\boxed{...}`` in the output (balanced
braces), else the last number in the output. Normalization: drop ``$``, ``,``, spaces and a trailing ``.``;
``\\frac``/``\\dfrac`` are not evaluated. A prediction matches when both sides parse as numbers and are equal
within 1e-6, or the normalized strings are equal. Output that hit ``max_new_tokens`` without a boxed answer is
scored like any other output (usually a miss).

Usage:
    python score_generation.py --task <prompts.jsonl> --results <results.jsonl> --out <scores.json> \\
        [--tokenizer <tokenizer.json>] [--baseline <results.jsonl>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
        i += 1
    return None


def extract(text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    nums = NUMBER.findall(text)
    return nums[-1] if nums else None


def normalize(value: str) -> str:
    v = value.replace("$", "").replace(",", "").replace(" ", "").replace("\\%", "").replace("%", "")
    v = re.sub(r"^\\text\{(.*)\}$", r"\1", v)
    return v.rstrip(".")


def matches(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gold)
    try:
        return abs(float(p) - float(g)) <= 1e-6
    except ValueError:
        return p == g


def outputs(path: Path, tokenizer: Path | None) -> dict[str, str]:
    rows = {r["id"]: r for r in _read(path)}
    tok = None
    out = {}
    for name, r in rows.items():
        if "output_text" in r:
            out[name] = r["output_text"]
            continue
        if tok is None:
            if tokenizer is None:
                raise SystemExit(f"{path}: rows carry output_ids only; pass --tokenizer")
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tokenizer))
        out[name] = tok.decode(r["output_ids"], skip_special_tokens=False)
    return out


def score(task: list[dict], texts: dict[str, str]) -> dict:
    missing = [t["id"] for t in task if t["id"] not in texts]
    if missing:
        raise SystemExit(f"results miss {len(missing)} task rows, e.g. {missing[:5]}")
    per = {t["id"]: matches(extract(texts[t["id"]]), t["answer"]) for t in task}
    return {"exact_match": sum(per.values()) / len(per), "correct": sum(per.values()), "rows": len(per), "per_row": per}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--baseline", type=Path, help="results.jsonl of the reference run, for per-row agreement")
    args = ap.parse_args()
    task = _read(args.task)
    result = score(task, outputs(args.results, args.tokenizer))
    result["task_sha256"] = hashlib.sha256(args.task.read_bytes()).hexdigest()
    result["results_sha256"] = hashlib.sha256(args.results.read_bytes()).hexdigest()
    if args.baseline:
        base = score(task, outputs(args.baseline, args.tokenizer))["per_row"]
        flips = sorted(k for k in base if base[k] != result["per_row"][k])
        result["vs_baseline"] = {"baseline_exact_match": sum(base.values()) / len(base), "flipped_rows": flips}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_row"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
