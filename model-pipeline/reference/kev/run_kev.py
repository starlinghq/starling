"""Stage runner for the kev-onnx-cpu-v1 onboarding route (model-pipeline/src/model_pipeline/onboarding_kev.py).

Runs in the Kev reference environment (reference/kev/requirements.txt plus Kev installed from a checkout at the
candidate's pinned code commit). It never imports model_pipeline: only the standard library, the reference packages
and the spike helpers next to this file (kev_merge.py, kev_ort.py).

Stages:
  reference  Kev's PyTorch fp32 path (Checkpoint.load on CPU, default LoadOptions) on a corpus of internal records.
             Writes reference.json (the contract the .NET fidelity tool reads) and reference-meta.json (environment,
             labels, the cached-state PyTorch layout, tokenizer cross-check).
  build      kev_merge.py (fp32 LoRA merge, head to safetensors) then the ONNX Runtime GenAI builder
             (-p fp32 -e cpu, exclude_lm_head, exclude_mtp). Writes bundle/ with exactly model.onnx, model.onnx.data,
             head.safetensors, head.json and tokenizer.json (the pinned Kev file), and build.json with their digests.
  onnx       Python ONNX Runtime (CPU EP) on the bundle: each question from the cached state, each question as one
             uncached causal row, and each multi-question record as one right-padded batch. Writes onnx.json.

Every Kev process must run with TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 (head.pt is a pickle) and offline Hugging Face access;
the base checkpoint is read from the Hugging Face cache that holds the pinned base snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE_FILES = ("model.onnx", "model.onnx.data", "head.safetensors", "head.json", "tokenizer.json")
DELIMITER_TOKENS = {
    "state": "<|fim_prefix|>",
    "question": "<|fim_middle|>",
    "option": "<|box_start|>",
    "option_end": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}
SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fail(message: str) -> None:
    raise SystemExit(f"kev runner: {message}")


def require_environment() -> None:
    if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") != "1":
        fail("set TORCH_FORCE_WEIGHTS_ONLY_LOAD=1: head.pt is a pickle and must load with weights_only=True")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or not os.environ.get("HF_HUB_CACHE"):
        fail("set HF_HUB_OFFLINE=1 and HF_HUB_CACHE to the cache holding the pinned base snapshot")


def load_profile(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def read_corpus(path: Path) -> list[dict]:
    records, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != {"id", "state", "questions"} or not isinstance(record["state"], str):
            fail(f"corpus line {number}: expected an internal record with id, state (string) and questions")
        if record["id"] in seen or not record["questions"]:
            fail(f"corpus line {number}: duplicate id or no questions")
        seen.add(record["id"])
        for question in record["questions"]:
            if set(question) != {"id", "instr", "options", "label"} or not question["options"]:
                fail(f"corpus line {number}: question needs id, instr, options (non-empty) and label")
            label = question["label"]
            if label is not None and (type(label) is not int or not 0 <= label < len(question["options"])):
                fail(f"corpus line {number}: label out of range")
        records.append(record)
    if not records:
        fail("empty corpus")
    return records


def check_backbone_config(model, behavior: dict) -> dict:
    config = model.lm.config
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    layer_types = list(text.layer_types)
    interval = behavior["full_attention_interval"]
    observed = {
        "hidden_size": text.hidden_size,
        "num_hidden_layers": text.num_hidden_layers,
        "num_key_value_heads": text.num_key_value_heads,
        "linear_num_key_heads": text.linear_num_key_heads,
        "linear_num_value_heads": text.linear_num_value_heads,
        "head_dim": text.head_dim,
    }
    for key, value in observed.items():
        if value != behavior[key]:
            fail(f"loaded backbone {key} is {value}, contract says {behavior[key]}")
    expected_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(text.num_hidden_layers)
    ]
    if layer_types != expected_types:
        fail("loaded backbone layer_types differ from the contract's full_attention_interval pattern")
    return {**observed, "layer_types": layer_types}


def cmd_reference(args) -> None:
    require_environment()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import SPECIAL as KEV_SPECIAL
    from kev.model import pad_id, rows_of, user_tokens
    from tokenizers import Tokenizer

    profile, profile_digest = load_profile(args.profile)
    behavior = profile["behavior"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    checkpoint = Checkpoint(str(args.snapshot))
    base = profile["origin"]["base"]
    if (checkpoint.meta.base, checkpoint.meta.base_revision) != (base["repo"], base["revision"]):
        fail(f"head.pt names base {checkpoint.meta.base}@{checkpoint.meta.base_revision}, contract {base}")
    if checkpoint.meta.option_isolation or checkpoint.meta.weights != "lora" or checkpoint.meta.weights_dtype != "fp32":
        fail("checkpoint is not an fp32 LoRA adapter without option isolation")
    started = time.perf_counter()
    tok, model = checkpoint.load("cpu", LoadOptions())  # fp32, LoRA merged in fp32, calibrated temperature
    load_seconds = time.perf_counter() - started
    if not model.hybrid:
        fail("contract requires the hybrid (Gated DeltaNet) Qwen3.5 backbone")
    temperature = float(model.head.temperature)
    if temperature != behavior["temperature"] or checkpoint.meta.head_dim != behavior["pointer_dim"]:
        fail(f"head.pt temperature {temperature!r} / dim {checkpoint.meta.head_dim} differ from the contract")
    config = check_backbone_config(model, behavior)
    delimiters = {name: tok.convert_tokens_to_ids(token) for name, token in DELIMITER_TOKENS.items()}
    if list(DELIMITER_TOKENS.values()) != list(KEV_SPECIAL) or delimiters != behavior["delimiters"]:
        fail(f"tokenizer delimiter ids {delimiters} differ from the contract")
    if pad_id(tok) != behavior["pad_token_id"]:
        fail(f"tokenizer pad id {pad_id(tok)} differs from the contract")
    # The pinned Kev tokenizer.json is what the .NET side loads; it must tokenize every segment exactly as the
    # tokenizer Kev builds at runtime from the base (transformers 5.17.0).
    pinned = Tokenizer.from_file(str(args.snapshot / behavior["tokenizer_file"]))

    def pinned_tokens(text: str) -> list[int]:
        return pinned.encode(SPECIAL.sub(r"<¦\1¦>", text), add_special_tokens=False).ids

    cases, meta_cases, segments = [], [], 0
    for record in read_corpus(args.corpus):
        enc = model.encode(
            tok, record, max_state=behavior["max_state_tokens"], max_branch=behavior["max_row_tokens"], strict=True
        )
        texts = [record["state"]] + [t for q in record["questions"] for t in (q["instr"], *q["options"])]
        for text in texts:
            segments += 1
            if user_tokens(tok, text) != pinned_tokens(text):
                fail(f"{record['id']}: pinned tokenizer.json differs from Kev's runtime tokenizer on {text!r}")
        state_ids, state_pos, rows = rows_of(enc)
        if state_pos != list(range(len(state_ids))) or any(
            r["pos"] != list(range(len(state_ids), len(state_ids) + len(r["ids"]))) for r in rows
        ):
            fail(f"{record['id']}: unexpected positions; the hybrid row form continues the state positions")
        with torch.no_grad():
            started = time.perf_counter()
            forward = [F.softmax(z, -1).double().cpu().numpy() for z in model.forward(enc)]
            forward_seconds = time.perf_counter() - started
            started = time.perf_counter()
            cached = [p.double().numpy() for p in model.probs(enc)]
            cached_seconds = time.perf_counter() - started
        for name, probs in (("forward", forward), ("probs", cached)):
            if not all(np.isfinite(p).all() for p in probs):
                fail(f"{record['id']}: non-finite PyTorch {name} probabilities")
        cases.append(
            {
                "id": record["id"],
                "record": {"state": record["state"], "questions": record["questions"]},
                "state_ids": [int(x) for x in state_ids],
                "rows": [
                    {
                        "ids": [int(x) for x in r["ids"]],
                        "pos": [int(x) for x in r["pos"]],
                        "decide": int(r["decide"]),
                        "opts": [int(x) for x in r["opts"]],
                    }
                    for r in rows
                ],
                "probs": [[float(x) for x in p] for p in forward],
            }
        )
        meta_cases.append(
            {
                "id": record["id"],
                "state_tokens": len(state_ids),
                "row_tokens": [len(state_ids) + len(r["ids"]) for r in rows],
                "labels": [q["label"] for q in record["questions"]],
                "probs_cached_layout": [[float(x) for x in p] for p in cached],
                "forward_vs_cached_layout_max_abs": max(
                    float(np.abs(a - b).max()) for a, b in zip(forward, cached, strict=True)
                ),
                "seconds": {"forward": forward_seconds, "cached": cached_seconds},
            }
        )
        print(record["id"], len(state_ids), meta_cases[-1]["forward_vs_cached_layout_max_abs"], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "reference.json", {"profile": profile["id"], "candidate_sha256": profile_digest, "cases": cases}
    )
    import kev
    import transformers

    write_json(
        args.out / "reference-meta.json",
        {
            "profile": profile["id"],
            "candidate_sha256": profile_digest,
            "reference_sha256": sha256(args.out / "reference.json"),
            "layout": "probs: kev DecisionModel.forward (state recomputed per question row, rows batched as Kev "
            "batches them), softmax with the calibrated temperature; probs_cached_layout: DecisionModel.probs",
            "environment": {
                "machine": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "kev_source": str(Path(kev.__file__).resolve().parent),
                "threads": args.threads,
                "device": "cpu",
                "load_options": "default (fp32, merge)",
                "load_seconds": load_seconds,
            },
            "checkpoint": {
                "base": checkpoint.meta.base,
                "base_revision": checkpoint.meta.base_revision,
                "temperature": temperature,
                "pointer_dim": checkpoint.meta.head_dim,
            },
            "backbone": config,
            "delimiters": delimiters,
            "pad_token_id": pad_id(tok),
            "tokenizer_cross_check": {"segments": segments, "status": "identical"},
            "cases": meta_cases,
        },
    )


def onnx_summary(path: Path) -> dict:
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    ops: dict[str, int] = {}
    for node in graph.node:
        key = f"{node.domain or 'ai.onnx'}::{node.op_type}"
        ops[key] = ops.get(key, 0) + 1

    def shape(value):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in value.type.tensor_type.shape.dim]

    return {
        "inputs": {value.name: shape(value) for value in graph.input},
        "outputs": {value.name: shape(value) for value in graph.output},
        "ops": dict(sorted(ops.items())),
    }


def cmd_build(args) -> None:
    require_environment()
    profile, profile_digest = load_profile(args.profile)
    behavior = profile["behavior"]
    tokenizer_pin = next(f for f in profile["origin"]["files"] if f["path"] == behavior["tokenizer_file"])
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = args.out / "bundle"
    bundle.mkdir()
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
    timings = {}
    with tempfile.TemporaryDirectory(prefix="build-", dir=args.out) as scratch:
        merged, built = Path(scratch) / "merged", Path(scratch) / "onnx"
        started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(HERE / "kev_merge.py"),
                "--checkpoint",
                str(args.snapshot),
                "--base",
                str(args.base_snapshot),
                "--out",
                str(merged),
            ],
            check=True,
            env=env,
        )
        timings["merge_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "onnxruntime_genai.models.builder",
                "-i",
                str(merged),
                "-o",
                str(built),
                "-p",
                "fp32",
                "-e",
                "cpu",
                "-c",
                str(Path(scratch) / "cache"),
                "--extra_options",
                "exclude_lm_head=true",
                "exclude_mtp=true",
            ],
            check=True,
            env=env,
        )
        timings["builder_seconds"] = time.perf_counter() - started
        genai_config = json.loads((built / "genai_config.json").read_text(encoding="utf-8"))
        shutil.move(built / "model.onnx", bundle / "model.onnx")
        shutil.move(built / "model.onnx.data", bundle / "model.onnx.data")
        shutil.move(merged / "head.safetensors", bundle / "head.safetensors")
        shutil.move(merged / "head.json", bundle / "head.json")
    shutil.copyfile(args.snapshot / behavior["tokenizer_file"], bundle / "tokenizer.json")
    if sha256(bundle / "tokenizer.json") != tokenizer_pin["sha256"]:
        fail("bundle tokenizer.json differs from the pinned Kev tokenizer")
    head = json.loads((bundle / "head.json").read_text(encoding="utf-8"))
    if (
        head["temperature"] != behavior["temperature"]
        or head["head_dim"] != behavior["pointer_dim"]
        or head["hidden_size"] != behavior["hidden_size"]
    ):
        fail(f"head.json {head} differs from the contract")
    summary = onnx_summary(bundle / "model.onnx")
    layers, interval = behavior["num_hidden_layers"], behavior["full_attention_interval"]
    attention = layers // interval
    expected_ops = {
        "com.microsoft::LinearAttention": layers - attention,
        "com.microsoft::CausalConvWithState": layers - attention,
        "com.microsoft::GroupQueryAttention": attention,
    }
    for op, count in expected_ops.items():
        if summary["ops"].get(op) != count:
            fail(f"graph has {summary['ops'].get(op)} {op} nodes, expected {count}")
    hidden = summary["outputs"].get("hidden_states")
    if not hidden or hidden[-1] != behavior["hidden_size"]:
        fail(f"graph hidden_states output {hidden} does not end in hidden size {behavior['hidden_size']}")
    import onnxruntime_genai

    write_json(
        args.out / "build.json",
        {
            "profile": profile["id"],
            "candidate_sha256": profile_digest,
            "bundle": {
                name: {"size": (bundle / name).stat().st_size, "sha256": sha256(bundle / name)} for name in BUNDLE_FILES
            },
            "builder": {
                "onnxruntime_genai": onnxruntime_genai.__version__,
                "arguments": "-p fp32 -e cpu --extra_options exclude_lm_head=true exclude_mtp=true",
                "context_length": genai_config.get("model", {}).get("context_length"),
            },
            "graph": summary,
            "head": head,
            "timings": timings,
        },
    )


def cmd_onnx(args) -> None:
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(HERE))
    from kev_ort import Backbone, Head, clone

    profile, profile_digest = load_profile(args.profile)
    behavior = profile["behavior"]
    build = json.loads(args.build.read_text(encoding="utf-8"))
    for name, entry in build["bundle"].items():
        if sha256(args.bundle / name) != entry["sha256"]:
            fail(f"bundle file {name} differs from build.json")
    reference_digest = sha256(args.reference)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference["profile"] != profile["id"] or reference["candidate_sha256"] != profile_digest:
        fail("reference.json names another candidate")
    backbone = Backbone(args.bundle / "model.onnx", args.threads, kv_dim=behavior["head_dim"], provider="cpu")
    head = Head(args.bundle)
    if float(head.temperature) != np.float32(behavior["temperature"]):
        fail("head.json temperature differs from the contract")
    pad = behavior["pad_token_id"]

    def probs(hidden_rows, readout) -> list[float]:
        selected = hidden_rows[readout]
        return [float(x) for x in head.softmax(head.logits(selected[0], selected[1:]))]

    cases = []
    started = time.perf_counter()
    for case in reference["cases"]:
        state = np.array(case["state_ids"], dtype=np.int64)
        n_state = len(state)
        full_rows, uncached, cached = [], [], []
        for row in case["rows"]:
            ids = np.concatenate([state, np.array(row["ids"], dtype=np.int64)])
            pos = np.concatenate([np.arange(n_state), np.array(row["pos"], dtype=np.int64)])
            readout = n_state + np.array([row["decide"], *row["opts"]])
            hidden, _ = backbone.run(ids[None], pos[None], np.ones((1, len(ids)), dtype=np.int64))
            uncached.append(probs(hidden[0], readout))
            full_rows.append((ids, pos, readout))
        hidden_state, present = backbone.run(
            state[None], np.arange(n_state)[None], np.ones((1, n_state), dtype=np.int64)
        )
        for row in case["rows"]:
            ids = np.array(row["ids"], dtype=np.int64)
            pos = np.array(row["pos"], dtype=np.int64)
            mask = np.ones((1, n_state + len(ids)), dtype=np.int64)
            hidden, _ = backbone.run(ids[None], pos[None], mask, clone(present))
            cached.append(probs(hidden[0], np.array([row["decide"], *row["opts"]])))
        batch = None
        if len(full_rows) > 1:
            width = max(len(ids) for ids, _, _ in full_rows)
            ids_b = np.full((len(full_rows), width), pad, dtype=np.int64)
            pos_b = np.zeros((len(full_rows), width), dtype=np.int64)
            mask_b = np.zeros((len(full_rows), width), dtype=np.int64)
            for i, (ids, pos, _) in enumerate(full_rows):
                ids_b[i, : len(ids)], pos_b[i, : len(ids)], mask_b[i, : len(ids)] = ids, pos, 1
            hidden, _ = backbone.run(ids_b, pos_b, mask_b)
            batch = {
                "row_lengths": [len(ids) for ids, _, _ in full_rows],
                "width": width,
                "output": [probs(hidden[i], readout) for i, (_, _, readout) in enumerate(full_rows)],
            }
        finite = bool(np.isfinite(hidden_state).all())
        cases.append(
            {
                "id": case["id"],
                "output": cached,
                "uncached_output": uncached,
                "batch": batch,
                "state_hidden_finite": finite,
            }
        )
        print(case["id"], n_state, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "onnx.json",
        {
            "profile": profile["id"],
            "candidate_sha256": profile_digest,
            "reference_sha256": reference_digest,
            "environment": {
                "machine": platform.node(),
                "onnxruntime": ort.__version__,
                "provider": "CPUExecutionProvider",
                "threads": args.threads,
                "python": platform.python_version(),
                "model_sha256": {name: entry["sha256"] for name, entry in build["bundle"].items()},
                "seconds": time.perf_counter() - started,
            },
            "cases": cases,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["reference", "build", "onnx"])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help="pinned Kev checkpoint directory")
    parser.add_argument("--base-snapshot", type=Path, help="pinned base snapshot directory (Hugging Face cache layout)")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--build", type=Path, help="build.json of the bundle")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        fail("threads must be 1 to 4")
    needed = {
        "reference": ("snapshot", "corpus"),
        "build": ("snapshot", "base_snapshot"),
        "onnx": ("reference", "bundle", "build"),
    }[args.stage]
    if any(getattr(args, name) is None for name in needed):
        fail(f"{args.stage} needs --{' --'.join(n.replace('_', '-') for n in needed)}")
    {"reference": cmd_reference, "build": cmd_build, "onnx": cmd_onnx}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
