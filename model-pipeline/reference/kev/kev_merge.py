"""Kev-0.8B spike, step 3: merge the LoRA in fp32 and save a checkpoint the ONNX Runtime GenAI builder accepts.

Kev's own loader merges the adapter into the base in fp32 (`Checkpoint.load` with default LoadOptions). This script
takes that merged text backbone and writes a full `Qwen3_5ForConditionalGeneration` checkpoint: every base tensor
(vision tower and MTP layer included, unused by Kev) upcast to fp32, with the language-model tensors replaced by the
merged ones. config.json records float32, so the builder does not round the merged weights back to bf16.

The pointer head goes to `head.safetensors` plus `head.json` (temperature, dimensions, source digest), so nothing
downstream unpickles `head.pt`.

Usage (Kev environment, TORCH_FORCE_WEIGHTS_ONLY_LOAD=1):
    python kev_merge.py --checkpoint <kev-0.8b dir> --base <Qwen3.5-0.8B-Base dir> --out <merged dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from kev.checkpoint import Checkpoint, LoadOptions
from safetensors import safe_open
from safetensors.torch import save_file

LM_PREFIX = "model.language_model."
COPY = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json", "LICENSE"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") != "1":
        raise SystemExit(
            "set TORCH_FORCE_WEIGHTS_ONLY_LOAD=1: head.pt is a pickle and must load with weights_only=True"
        )

    ck = Checkpoint(str(args.checkpoint))
    _, model = ck.load("cpu", LoadOptions())  # fp32, LoRA merged (merge_and_unload)
    merged = {k: v.detach().to(torch.float32).contiguous() for k, v in model.lm.state_dict().items()}

    tensors: dict[str, torch.Tensor] = {}
    replaced = 0
    shards = sorted(args.base.glob("model.safetensors-*.safetensors"))
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open exposes keys(), not iteration
                if key.startswith(LM_PREFIX) and key[len(LM_PREFIX) :] in merged:
                    tensors[key] = merged[key[len(LM_PREFIX) :]]
                    replaced += 1
                else:
                    tensors[key] = f.get_tensor(key).to(torch.float32).contiguous()
    lm_keys = [k for k in tensors if k.startswith(LM_PREFIX)]
    if replaced != len(merged) or replaced != len(lm_keys):
        raise SystemExit(
            f"merged {len(merged)} tensors, replaced {replaced}, base has {len(lm_keys)} language-model tensors"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    del tensors

    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    config["dtype"] = "float32"
    for sub in ("text_config", "vision_config"):
        if sub in config:
            config[sub]["dtype"] = "float32"
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in COPY:
        if (args.base / name).exists():
            shutil.copy2(args.base / name, args.out / name)

    head = {k: v.detach().to(torch.float32).contiguous() for k, v in model.head.state_dict().items()}
    save_file(head, str(args.out / "head.safetensors"))
    meta = {
        "temperature": float(model.head.temperature),
        "head_dim": int(head["q.weight"].shape[0]),
        "hidden_size": int(head["q.weight"].shape[1]),
        "logits": "(k(h_opts) @ q(h_decide)) / sqrt(head_dim) / temperature",
        "head_pt_sha256": sha256(args.checkpoint / "head.pt"),
        "adapter_sha256": sha256(args.checkpoint / "adapter_model.safetensors"),
        "base_revision": ck.meta.base_revision,
    }
    (args.out / "head.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"replaced": replaced, "files": sorted(p.name for p in args.out.iterdir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
