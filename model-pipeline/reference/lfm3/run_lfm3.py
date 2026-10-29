"""Pinned LFM3-1.7B reference (transformers trust_remote_code, PyTorch fp32 on CPU) and the pinned ONNX export on
ONNX Runtime, into proposed artifacts only.

template:  for each corpus case, the checkpoint's chat template rendered by transformers apply_chat_template
           (add_generation_prompt, enable_thinking and preserve_thinking as the case gives) and the prompt ids of the
           checkpoint's tokenizer (add_special_tokens=False).
reference: greedy generation with the pinned checkpoint in fp32: do_sample False and repetition_penalty 1.0 (the
           generation config's sampling defaults overridden), eos <|im_end|>; the generated ids and the step-0 logits.
onnx:      the pinned step graph on ONNX Runtime, CPU, an independent greedy loop over the same prompt ids
           (logits_to_keep 1, caches carried between calls); the generated ids and the step-0 logits.

Writes <out>/template.jsonl, <out>/<stage>.jsonl and <out>/<stage>-step0/<id>.f32 (little-endian float32, 128,000
values). Nothing is written into protected/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

CHECKPOINT = ("LiquidAI/LFM3-1.7B", "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd")
ONNX_EXPORT = ("LiquidAI/LFM3-1.7B-ONNX", "c680759afee1ce8310c90d6c621f4558200dee77")
IM_END = 124900
# config.json layer_types: 13 conv, 6 full_attention and 5 sliding_attention layers.
LAYER_TYPES = ["conv", "conv", *["full_attention", "conv", "sliding_attention", "conv"] * 5, "full_attention", "conv"]


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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render(tokenizer, case: dict) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        case["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=case.get("enable_thinking", True),
        preserve_thinking=case.get("preserve_thinking", False),
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, ids


def write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(path)


def run_reference(checkpoint: Path, prompts: dict[str, list[int]], cases: list[dict], out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    rows = []
    for case in cases:
        ids = prompts[case["id"]]
        config = GenerationConfig(
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=case["max_new_tokens"],
            eos_token_id=[IM_END],
            pad_token_id=124893,
            output_logits=True,
            return_dict_in_generate=True,
        )
        with torch.inference_mode():
            result = model.generate(torch.tensor([ids]), generation_config=config)
        generated = result.sequences[0, len(ids) :].tolist()
        write_f32(out / "reference-step0" / f"{case['id']}.f32", result.logits[0][0].float().numpy())
        rows.append({"id": case["id"], "generated_ids": generated})
        print(case["id"], len(ids), generated[:8], flush=True)
    (out / "reference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def run_onnx(model_path: Path, prompts: dict[str, list[int]], cases: list[dict], out: Path, threads: int) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    rows = []
    for case in cases:
        ids = prompts[case["id"]]
        past = {}
        for layer, kind in enumerate(LAYER_TYPES):
            if kind == "conv":
                past[f"past_conv.{layer}"] = np.zeros((1, 2048, 7), dtype=np.float32)
            else:
                past[f"past_key_values.{layer}.key"] = np.zeros((1, 4, 0, 128), dtype=np.float32)
                past[f"past_key_values.{layer}.value"] = np.zeros((1, 4, 0, 128), dtype=np.float32)
        names = [o.name for o in session.get_outputs()]
        step_ids = list(ids)
        seen = 0
        generated: list[int] = []
        for step in range(case["max_new_tokens"]):
            n = len(step_ids)
            feeds = {
                "input_ids": np.array([step_ids], dtype=np.int64),
                "attention_mask": np.ones((1, seen + n), dtype=np.int64),
                "position_ids": np.arange(seen, seen + n, dtype=np.int64)[None, :],
                "logits_to_keep": np.array([1], dtype=np.int64),
                **past,
            }
            outputs = dict(zip(names, session.run(names, feeds), strict=True))
            logits = outputs["logits"][0, -1]
            if step == 0:
                write_f32(out / "onnx-step0" / f"{case['id']}.f32", logits)
            seen += n
            for key in list(past):
                past[key] = outputs[key.replace("past_conv.", "present_conv.").replace("past_key_values.", "present.")]
            token = int(np.argmax(logits))
            if token == IM_END:
                break
            generated.append(token)
            step_ids = [token]
        rows.append({"id": case["id"], "generated_ids": generated})
        print(case["id"], len(ids), generated[:8], flush=True)
    (out / "onnx.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["template", "reference", "onnx"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="LiquidAI/LFM3-1.7B snapshot (pinned)")
    parser.add_argument("--onnx-model", type=Path, help="onnx/model.onnx of the pinned export")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    cases = load_corpus(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    args.out.mkdir(parents=True, exist_ok=True)
    prompts: dict[str, list[int]] = {}
    template_rows = []
    for case in cases:
        text, ids = render(tokenizer, case)
        prompts[case["id"]] = ids
        template_rows.append({**case, "text": text, "prompt_ids": ids})

    if args.stage == "template":
        (args.out / "template.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in template_rows), encoding="utf-8"
        )
    elif args.stage == "reference":
        run_reference(args.checkpoint, prompts, cases, args.out)
    else:
        if args.onnx_model is None:
            raise SystemExit("--onnx-model is required for the onnx stage")
        import onnxruntime

        run_onnx(args.onnx_model, prompts, cases, args.out, args.threads)

    environment = {
        "stage": args.stage,
        "checkpoint": {"repo": CHECKPOINT[0], "revision": CHECKPOINT[1]},
        "onnx_export": {"repo": ONNX_EXPORT[0], "revision": ONNX_EXPORT[1]},
        "chat_template_sha256": sha256(args.checkpoint / "chat_template.jinja"),
        "tokenizer_sha256": sha256(args.checkpoint / "tokenizer.json"),
        "corpora": [{"path": str(p), "sha256": sha256(p)} for p in args.corpus],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    if args.stage == "onnx":
        environment["packages"]["onnxruntime"] = onnxruntime.__version__
        environment["onnx_model_sha256"] = sha256(args.onnx_model)
    (args.out / f"environment-{args.stage}.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
