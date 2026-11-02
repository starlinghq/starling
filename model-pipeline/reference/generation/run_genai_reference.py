"""Greedy-generation reference of a built ONNX Runtime GenAI graph through the Python binding (issue #163).

The transformers fp32 reference (``run_reference.py``) is the reference of the fp32x and fp16x profiles. A quantized
graph cannot match it (the int4 CPU build agrees with it on 1 of 22 prompts, ``docs/m0/generation-backend-check.md``
section 3), so a quantized profile is compared with the same builder output run by the upstream Python binding
(onnxruntime-genai 0.17.1 from pip): this runner. Two steps, in separate processes:

1. ``run``: feeds the prompt ids of an accepted fixture (the prompt side stays the shared, accepted one) to the graph
   and writes the shared result format of ``compare.py``:

   - ``<out>/results.jsonl``: per prompt ``id``, ``backend``, ``rendered`` (copied from the fixture), ``prompt_ids``,
     ``output_ids`` (with the stop token when one was generated), ``finish_reason``, ``step_logprob``, ``step_margin``,
     ``top5_step0``, ``output_text`` (empty until step 2), ``elapsed_s``;
   - ``<out>/step0/<id>.f32``: step-0 log-softmax over the full vocabulary, float32 little-endian;
   - ``<out>/environment.json``: packages, graph digests, overlays, threads.

2. ``annotate``: loads only the snapshot's transformers tokenizer (no GenAI), re-renders every prompt with the chat
   template and checks the rendering and ids against the fixture, decodes ``output_text``
   (``skip_special_tokens=False``) and records the check in ``environment.json``. It is a separate process because a
   ``run`` with transformers loaded was killed twice (SIGKILL) at the 1,666-token prompt on the shared host, while the
   same ``run`` without it completed.

Runtime settings equal the C# runtime's (``Qwen35ChatModel``): CPU provider, ``pad_token_id`` overlaid to 248319 (an
embedding row the tokenizer never produces), greedy search, ``max_length`` = prompt + ``max_new_tokens``, the prompt in
one ``append_tokens`` call up to 1,024 tokens and 1,024-token calls above (the C# CPU default; ``--prefill-chunk``
changes it, 0 is one call). Each step's logits are read after ``generate_next_token`` (the logits that selected the
token); the step-0 vector must equal the one after ``append_tokens`` and the selected token must be the argmax.

Run in the environment of ``requirements.txt``::

    python run_genai_reference.py run --graph <bundle>/ortgenai-int4 --fixture <generation.jsonl> \\
        --prompts prompts-v1.jsonl --out <dir> --threads 6
    python run_genai_reference.py annotate --results <dir> --snapshot <Qwen3.5-0.8B snapshot> \\
        --fixture <generation.jsonl> --prompts prompts-v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np

PAD_TOKEN_ID_OVERLAY = 248319
STOP_IDS = (248044, 248046)  # <|endoftext|>, <|im_end|>
VOCAB = 248320
GENAI_VERSION = "0.17.1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    m = x.max()
    return (x - m - np.log(np.exp(x - m).sum())).astype("<f4")


def last_logits(generator) -> np.ndarray:
    out = generator.get_output("logits")
    return np.array(out.reshape(-1, VOCAB)[-1], dtype=np.float32, copy=True)


def greedy(og, model, ids: list[int], max_new: int, chunk: int | None) -> dict:
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(ids) + max_new)
    generator = og.Generator(model, params)
    try:
        t0 = time.perf_counter()
        size = chunk or len(ids)
        for start in range(0, len(ids), size):
            generator.append_tokens(ids[start : start + size])
        prefill = last_logits(generator)
        gen: list[int] = []
        logprobs: list[float] = []
        margins: list[float] = []
        step0 = None
        for step in range(max_new):
            if generator.is_done():
                break
            generator.generate_next_token()
            token = int(generator.get_next_tokens()[0])
            logits = last_logits(generator)
            if step == 0 and not np.array_equal(logits, prefill):
                raise RuntimeError("step-0 logits after generate_next_token differ from those after append_tokens")
            top2 = np.argsort(-logits, kind="stable")[:2]
            if int(top2[0]) != token and logits[top2[0]] != logits[token]:
                raise RuntimeError(f"step {step}: greedy token {token} is not the argmax {int(top2[0])}")
            lp = log_softmax(logits)
            if step == 0:
                step0 = lp
            logprobs.append(float(lp[token]))
            margins.append(float(logits[top2[0]] - logits[top2[1]]))
            gen.append(token)
            if token in STOP_IDS:
                break
        elapsed = time.perf_counter() - t0
    finally:
        del generator
        del params
    order = np.argsort(-step0, kind="stable")[:5]
    return {
        "output_ids": gen,
        "finish_reason": "eos" if gen and gen[-1] in STOP_IDS else "length",
        "step_logprob": logprobs,
        "step_margin": margins,
        "top5_step0": [[int(i), float(step0[i])] for i in order],
        "elapsed_s": elapsed,
        "_step0": step0,
    }


def load_cases(fixture: Path, prompts_path: Path) -> tuple[list[dict], dict[str, dict]]:
    cases = read_jsonl(fixture)
    prompts = {p["id"]: p for p in read_jsonl(prompts_path)}
    if sorted(prompts) != sorted(c["id"] for c in cases):
        raise SystemExit("the fixture and the prompts file have different ids")
    return cases, prompts


def run(args: argparse.Namespace) -> int:
    if metadata.version("onnxruntime-genai") != GENAI_VERSION:
        raise SystemExit(f"onnxruntime-genai {GENAI_VERSION} is required (the builder version of the graph)")
    import onnxruntime_genai as og

    cases, _ = load_cases(args.fixture, args.prompts)
    overlays = [
        {"model": {"decoder": {"session_options": {"intra_op_num_threads": args.threads}}}},
        {"model": {"pad_token_id": PAD_TOKEN_ID_OVERLAY}},
    ]
    config = og.Config(str(args.graph))
    config.clear_providers()
    for overlay in overlays:
        config.overlay(json.dumps(overlay))
    model = og.Model(config)

    chunk = args.prefill_chunk or None
    (args.out / "step0").mkdir(parents=True, exist_ok=True)
    with (args.out / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            ids = list(case["prompt_ids"])
            r = greedy(og, model, ids, case["max_new_tokens"], chunk)
            r.pop("_step0").tofile(args.out / "step0" / f"{case['id']}.f32")
            row = {
                "id": case["id"],
                "backend": f"onnxruntime-genai-python-cpu-{args.graph.name}",
                "rendered": case["rendered"],
                "prompt_ids": ids,
                **r,
                "output_text": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"{case['id']:<20} prompt {len(ids):>5}  gen {len(r['output_ids']):>4}  {r['finish_reason']:<6} "
                f"{r['elapsed_s']:.2f}s",
                flush=True,
            )

    env = {
        "runner": "model-pipeline/reference/generation/run_genai_reference.py",
        "graph_dir": str(args.graph),
        "graph": {p.name: sha256_file(p) for p in sorted(args.graph.iterdir()) if p.is_file()},
        "provider": "cpu",
        "device_name": platform.processor() or platform.machine(),
        "intra_op_num_threads": args.threads,
        "prefill_chunk": chunk,
        "config_overlays": overlays,
        "search": {"do_sample": False, "max_length": "len(prompt_ids) + max_new_tokens"},
        "stop_ids": list(STOP_IDS),
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ("onnxruntime-genai", "onnxruntime", "numpy")},
        "prompts_sha256": sha256_file(args.prompts),
        "prompt_ids_from": str(args.fixture),
        "prompt_ids_from_sha256": sha256_file(args.fixture),
        "annotation": None,
    }
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    return 0


def annotate(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    cases, prompts = load_cases(args.fixture, args.prompts)
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    if len(tok) > PAD_TOKEN_ID_OVERLAY:
        raise SystemExit("the pad overlay must be an embedding row the tokenizer never produces")
    rows = {r["id"]: r for r in read_jsonl(args.results / "results.jsonl")}
    for case in cases:
        rec = prompts[case["id"]]
        rendered = tok.apply_chat_template(
            rec["messages"],
            tools=rec.get("tools"),
            add_generation_prompt=True,
            tokenize=False,
            **dict(rec.get("chat_template_kwargs") or {}),
        )
        if rendered != case["rendered"] or tok(rendered, add_special_tokens=False)["input_ids"] != case["prompt_ids"]:
            raise SystemExit(f"{case['id']}: the snapshot's rendering or ids differ from the fixture")
        row = rows[case["id"]]
        if row["prompt_ids"] != case["prompt_ids"]:
            raise SystemExit(f"{case['id']}: the run fed other prompt ids than the fixture's")
        row["output_text"] = tok.decode(row["output_ids"], skip_special_tokens=False)
    write_jsonl(args.results / "results.jsonl", [rows[c["id"]] for c in cases])
    env_path = args.results / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8"))
    env["annotation"] = {
        "prompt_ids_checked_against_snapshot": True,
        "output_text": "transformers tokenizer decode, skip_special_tokens=False",
        "snapshot_files": {
            p.name: sha256_file(p)
            for p in sorted(args.snapshot.iterdir())
            if p.is_file() and p.suffix in {".json", ".jinja", ".txt"} and p.name != "hashes.json"
        },
        "packages": {p: metadata.version(p) for p in ("transformers", "tokenizers")},
    }
    env_path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"annotated {len(cases)} rows; renderings and prompt ids equal the fixture")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="greedy generation through the Python binding")
    r.add_argument("--graph", required=True, type=Path, help="directory with genai_config.json, model.onnx(.data)")
    r.add_argument("--fixture", required=True, type=Path, help="accepted generation.jsonl whose prompt ids are fed")
    r.add_argument("--prompts", required=True, type=Path, help="the prompts file of that fixture (digest recorded)")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--threads", type=int, required=True, help="intra_op_num_threads (recorded)")
    r.add_argument(
        "--prefill-chunk",
        type=int,
        default=1024,
        help="prompt tokens per append_tokens call (default 1024, the C# CPU default; 0: one call)",
    )
    a = sub.add_parser("annotate", help="check renderings and ids against the snapshot, decode output text")
    a.add_argument("--results", required=True, type=Path, help="a run directory")
    a.add_argument("--snapshot", required=True, type=Path, help="Qwen/Qwen3.5-0.8B snapshot (its tokenizer)")
    a.add_argument("--fixture", required=True, type=Path)
    a.add_argument("--prompts", required=True, type=Path)
    args = ap.parse_args(argv)
    return run(args) if args.command == "run" else annotate(args)


if __name__ == "__main__":
    raise SystemExit(main())
