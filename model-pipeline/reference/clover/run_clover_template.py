"""Pinned Clover-35B-A3B-Preview template reference: for each corpus case, the checkpoint's chat template rendered by
transformers apply_chat_template and the prompt ids of the checkpoint's tokenizer (add_special_tokens=False).

A case passes the template's switches only when it names them (enable_thinking, preserve_thinking,
add_non_thinking_prefix), so the template's own defaults are exercised too; add_generation_prompt defaults to true.

Writes <out>/template.jsonl and <out>/environment-template.json. Nothing is written into protected/. The model itself
does not run: no torch, no remote code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

CHECKPOINT = ("Diffuspace/Clover-35B-A3B-Preview", "4b0402b65e523d0183a14c431955438714cb16ff")
SWITCHES = ("enable_thinking", "preserve_thinking", "add_non_thinking_prefix")


def load_corpus(paths: list[Path]) -> list[dict]:
    cases: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case ids")
    return cases


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="snapshot with tokenizer files and template")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    rows = []
    for case in load_corpus(args.corpus):
        kwargs = {k: case[k] for k in SWITCHES if k in case}
        text = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=case.get("add_generation_prompt", True),
            **kwargs,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        rows.append({**case, "text": text, "prompt_ids": ids})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "template.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    environment = {
        "stage": "template",
        "checkpoint": {"repo": CHECKPOINT[0], "revision": CHECKPOINT[1]},
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": sha256(args.checkpoint / "chat_template.jinja"),
        "tokenizer_sha256": sha256(args.checkpoint / "tokenizer.json"),
        "tokenizer_config_sha256": sha256(args.checkpoint / "tokenizer_config.json"),
        "corpora": [{"path": p.as_posix(), "sha256": sha256(p)} for p in args.corpus],
        "cases": len(rows),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
    }
    (args.out / "environment-template.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
