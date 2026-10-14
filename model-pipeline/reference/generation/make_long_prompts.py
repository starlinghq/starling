"""Long-context prompts for the pilot LLM (capability ``long_context.over_1666_tokens``).

Builds ``prompts-long-v1.jsonl`` in the ``prompts-v1.jsonl`` format: one user message per prompt holding a prefix of a
fixed document (repository documentation at a pinned commit, cut at a line boundary) followed by one question, sized
so the rendered prompt is just above each target length in tokens of the pinned tokenizer. The prompts carry their
full text, so the file is self-contained; this script only documents how it was made.

    python make_long_prompts.py --model-dir <snapshot> --out prompts-long-v1.jsonl \\
        --source docs/plan/fast-faithful-ai-dotnet.md --source docs/m0/generation-backend-check.md ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

QUESTION = "\n\nQuestion: In two sentences, what is the document above about?"
INTRO = "Read the following document.\n\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", action="append", required=True, type=Path)
    ap.add_argument("--targets", default="4096,8192,16384")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    lines: list[str] = []
    for src in args.source:
        lines += src.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        lines.append("")

    def prompt_len(n_lines: int) -> int:
        content = INTRO + "\n".join(lines[:n_lines]).strip() + QUESTION
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        return len(tok(rendered, add_special_tokens=False)["input_ids"])

    rows = []
    for target in (int(t) for t in args.targets.split(",")):
        lo, hi = 1, len(lines)
        if prompt_len(hi) < target:
            raise SystemExit(f"sources give only {prompt_len(hi)} tokens; target {target}")
        while lo < hi:  # smallest line count whose prompt reaches the target
            mid = (lo + hi) // 2
            if prompt_len(mid) >= target:
                hi = mid
            else:
                lo = mid + 1
        content = INTRO + "\n".join(lines[:lo]).strip() + QUESTION
        rows.append(
            {
                "id": f"long-{target}",
                "category": "long-context",
                "messages": [{"role": "user", "content": content}],
                "max_new_tokens": args.max_new_tokens,
            }
        )
        print(f"long-{target}: {lo} lines, {prompt_len(lo)} tokens, {len(content)} characters")

    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
