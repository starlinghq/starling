"""Python GenAI CPU path; compare full/chunked prefills and fresh request isolation."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path

from common import corpus_rows, logits_record, row_record, write_json


def generate(og, model, ids, maximum, eos, chunk=None):
    parameters = og.GeneratorParams(model)
    parameters.set_search_options(do_sample=False, max_length=len(ids) + maximum)
    generator = og.Generator(model, parameters)
    try:
        for start in range(0, len(ids), chunk or len(ids)):
            generator.append_tokens(ids[start : start + (chunk or len(ids))])
        initial = generator.get_output("logits").reshape(-1, 151936)[-1].copy()
        tokens, stop = [], None
        for _ in range(maximum):
            if generator.is_done():
                break
            generator.generate_next_token()
            token = int(generator.get_next_tokens()[0])
            if token in eos:
                stop = token
                break
            tokens.append(token)
        return tokens, stop, initial
    finally:
        del generator
        del parameters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "bundle", "snapshot", "corpus", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import onnxruntime_genai as og
    from transformers import AutoTokenizer

    if metadata.version("onnxruntime-genai") != "0.17.1":
        raise ValueError("GenAI builder/runtime must be 0.17.1")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    config = og.Config(str(args.bundle))
    config.clear_providers()
    raw_tokenizer = json.loads((args.snapshot / "tokenizer.json").read_text(encoding="utf-8"))
    ids = set(raw_tokenizer["model"]["vocab"].values()) | {token["id"] for token in raw_tokenizer["added_tokens"]}
    if 151935 in ids:
        raise ValueError("pad overlay must be an unused embedding row")
    config.overlay(
        json.dumps(
            {"model": {"pad_token_id": 151935, "decoder": {"session_options": {"intra_op_num_threads": args.threads}}}}
        )
    )
    model = og.Model(config)
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
            tokens, stop, logits = generate(og, model, ids, maximum, candidate["behavior"]["eos_token_ids"])
            chunked, chunked_stop, _ = generate(
                og, model, ids, maximum, candidate["behavior"]["eos_token_ids"], chunk=7
            )
            # Run a different request in between repeated original requests, sharing weights only.
            distractor = tokenizer.encode(
                "<|im_start|>user\nAnother request.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
            )
            generate(og, model, distractor, 1, candidate["behavior"]["eos_token_ids"])
            repeated, repeated_stop, _ = generate(og, model, ids, maximum, candidate["behavior"]["eos_token_ids"])
            row = row_record(
                candidate,
                record,
                "python-genai",
                text,
                ids,
                tokens,
                stop,
                tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                logits_record(args.out, index, logits, 151936),
                {
                    "cache": tokens == chunked and stop == chunked_stop,
                    "isolation": tokens == repeated and stop == repeated_stop,
                },
            )
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"{record['id']}: prompt={len(ids)} completion={len(tokens)} stop={stop}", flush=True)
    write_json(
        args.out / "environment.json",
        {
            "backend": "python-genai",
            "version": metadata.version("onnxruntime-genai"),
            "provider": "CPUExecutionProvider",
            "threads": args.threads,
            "pad_token_id_overlay": 151935,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
