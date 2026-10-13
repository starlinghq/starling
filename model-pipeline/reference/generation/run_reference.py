"""Greedy-generation reference for the pilot LLM (M0 generation-backend check).

Runs transformers in fp32 with the model's own chat template on ``prompts-v1.jsonl`` and writes, per device:

- ``<out>/results.jsonl``: one record per prompt in the shared result format (see ``compare.py``);
- ``<out>/step0/<id>.f32``: step-0 log-softmax over the full vocabulary, float32 little-endian;
- ``<out>/environment.json``: packages, device, model revision and asset hashes.

Greedy decoding stops at ``<|im_end|>`` or ``<|endoftext|>`` (the ids are read from the tokenizer) or at the
prompt's ``max_new_tokens``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def render(tok, rec: dict) -> tuple[str, list[int]]:
    kwargs = dict(rec.get("chat_template_kwargs") or {})
    rendered = tok.apply_chat_template(
        rec["messages"], tools=rec.get("tools"), add_generation_prompt=True, tokenize=False, **kwargs
    )
    ids_direct = tok.apply_chat_template(
        rec["messages"], tools=rec.get("tools"), add_generation_prompt=True, tokenize=True, return_dict=False, **kwargs
    )
    if isinstance(ids_direct, dict) or hasattr(ids_direct, "input_ids"):
        ids_direct = ids_direct["input_ids"]
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    if list(ids_direct) != list(ids):
        raise RuntimeError(f"{rec['id']}: apply_chat_template(tokenize=True) differs from tokenizing the rendered text")
    return rendered, list(ids)


@torch.inference_mode()
def greedy(model, ids: list[int], max_new: int, stop_ids: set[int], device: str) -> dict:
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    out = model.generate(
        inp,
        attention_mask=torch.ones_like(inp),
        do_sample=False,
        max_new_tokens=max_new,
        eos_token_id=sorted(stop_ids),
        pad_token_id=min(stop_ids),
        output_logits=True,
        return_dict_in_generate=True,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gen = out.sequences[0, len(ids) :].tolist()
    logprobs, margins, top5_0, step0 = [], [], None, None
    for step, logits in enumerate(out.logits):
        lp = torch.log_softmax(logits[0].float(), dim=-1)
        top = torch.topk(logits[0].float(), 2)
        logprobs.append(float(lp[gen[step]]))
        margins.append(float(top.values[0] - top.values[1]))
        if step == 0:
            step0 = lp.cpu().numpy().astype("<f4")
            t5 = torch.topk(lp, 5)
            top5_0 = [[int(i), float(v)] for v, i in zip(t5.values.tolist(), t5.indices.tolist(), strict=True)]
    finish = "eos" if gen and gen[-1] in stop_ids else "length"
    return {
        "output_ids": gen,
        "finish_reason": finish,
        "step_logprob": logprobs,
        "step_margin": margins,
        "top5_step0": top5_0,
        "elapsed_s": elapsed,
        "_step0": step0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--revision", required=True, help="commit SHA of the snapshot in --model-dir (recorded only)")
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    stop_ids = {tok.convert_tokens_to_ids(t) for t in STOP_TOKENS}
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).to(args.device).eval()

    prompts = load_prompts(args.prompts)
    out_dir = args.out
    (out_dir / "step0").mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in prompts:
            rendered, ids = render(tok, rec)
            r = greedy(model, ids, rec["max_new_tokens"], stop_ids, args.device)
            r.pop("_step0").tofile(out_dir / "step0" / f"{rec['id']}.f32")
            row = {
                "id": rec["id"],
                "backend": f"transformers-{args.device}-fp32",
                "rendered": rendered,
                "prompt_ids": ids,
                **r,
                "output_text": tok.decode(r["output_ids"], skip_special_tokens=False),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{rec['id']:<20} prompt {len(ids):>5}  gen {len(r['output_ids']):>4}  {r['finish_reason']:<6} "
                f"{r['elapsed_s']:.2f}s"
            )

    pkgs = {p: metadata.version(p) for p in ("torch", "transformers", "tokenizers", "numpy", "safetensors")}
    try:
        pkgs["flash-linear-attention"] = metadata.version("flash-linear-attention")
    except metadata.PackageNotFoundError:
        pkgs["flash-linear-attention"] = None
    env = {
        "runner": "model-pipeline/reference/generation/run_reference.py",
        "model_dir": str(args.model_dir),
        "revision": args.revision,
        "model_class": type(model).__name__,
        "dtype": "float32",
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "tf32": False,
        "stop_ids": sorted(stop_ids),
        "python": platform.python_version(),
        "packages": pkgs,
        "prompts_sha256": sha256_file(args.prompts),
        "assets": {
            p.name: sha256_file(p)
            for p in sorted(args.model_dir.iterdir())
            if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".safetensors"} and p.name != "hashes.json"
        },
    }
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
