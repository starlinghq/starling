"""Original transformers fp32 CPU reference; outputs are proposals, never protected fixtures."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path

from common import corpus_rows, logits_record, row_record, write_json

from model_pipeline.evaluate_generation_candidate import TRANSFORMERS_VERSION


def greedy(model, torch, ids: list[int], maximum: int, eos: list[int], *, cached: bool):
    generated, past, initial = [], None, None
    for step in range(maximum):
        # Cached prefill uses the full prompt; subsequent calls use one selected token.
        current = ids + generated if not cached or step == 0 else [generated[-1]]
        arguments = {"input_ids": torch.tensor([current], dtype=torch.long), "use_cache": cached}
        if cached and past is not None:
            arguments["past_key_values"] = past
        with torch.inference_mode():
            output = model(**arguments)
        logits = output.logits[0, -1].float()
        if step == 0:
            initial = logits.cpu().numpy().copy()
        token = int(torch.argmax(logits))
        if token in eos:
            return generated, token, initial
        generated.append(token)
        if cached:
            past = output.past_key_values
    return generated, None, initial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "snapshot", "corpus", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError("reference/tokenizer version differs from the admitted 5.18.0 environment")

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot,
        dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    if type(model).__name__ != candidate["behavior"]["architecture"] or model.config.vocab_size != 151936:
        raise ValueError("model class/vocabulary differs from candidate")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.jsonl").open("w", encoding="utf-8") as stream:
        for index, record in enumerate(corpus_rows(args.corpus)):
            text = tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            maximum = record.get("max_new_tokens", candidate["behavior"]["max_new_tokens"])
            if type(maximum) is not int or maximum < 1:
                raise ValueError("max_new_tokens must be positive")
            tokens, stop, logits = greedy(
                model, torch, ids, maximum, candidate["behavior"]["eos_token_ids"], cached=True
            )
            uncached, uncached_stop, _ = greedy(
                model,
                torch,
                ids,
                maximum,
                candidate["behavior"]["eos_token_ids"],
                cached=False,
            )
            row = row_record(
                candidate,
                record,
                "transformers",
                text,
                ids,
                tokens,
                stop,
                tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                logits_record(args.out, index, logits, 151936),
                {"cache": tokens == uncached and stop == uncached_stop},
            )
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"{record['id']}: prompt={len(ids)} completion={len(tokens)} stop={stop}", flush=True)
    write_json(
        args.out / "environment.json",
        {
            "reference_status": "proposed",
            "automatically_adopted": False,
            "threads": args.threads,
            "packages": {name: metadata.version(name) for name in ("torch", "transformers", "tokenizers", "numpy")},
            "architecture": type(model).__name__,
            "dtype": "float32",
            "attention": "eager",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
