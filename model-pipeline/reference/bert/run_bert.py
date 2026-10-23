"""Pinned BERT original-framework reference or ONNX execution, into proposed artifacts only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tokenizers import Tokenizer

from model_pipeline import fixtures
from model_pipeline.onboarding_bert import CONTRACTS, validate_candidate_behavior


def pooled(hidden: np.ndarray, mask: np.ndarray, mode: str, normalize: str) -> np.ndarray:
    if mode == "cls":
        values = hidden[:, 0]
    else:
        expanded = mask[..., None].astype(np.float32)
        values = (hidden * expanded).sum(axis=1) / np.clip(expanded.sum(axis=1), 1e-9, None)
    if normalize == "l2":
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    return values


def load_cases(path: Path, candidate: dict, tokenizer: Tokenizer) -> list[dict]:
    rows = fixtures.read_jsonl(path)
    out, seen = [], set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("corpus cases require unique nonempty ids")
        seen.add(identifier)
        if candidate["task"] == "embeddings":
            text, pair, prompt = row["text"], None, row.get("prompt", "none")
            prefix = candidate["behavior"]["prompts"][prompt]
        else:
            text, pair, prompt, prefix = row["query"], row["document"], "none", ""
        if not isinstance(text, str) or (pair is not None and not isinstance(pair, str)):
            raise ValueError(f"{identifier}: corpus text/pair must be strings")
        encoding = tokenizer.encode(prefix + text, pair) if pair is not None else tokenizer.encode(prefix + text)
        out.append(
            {
                "id": identifier,
                "text": text,
                "pair": pair,
                "prompt": prompt,
                "input_ids": encoding.ids,
                "attention_mask": encoding.attention_mask,
                "token_type_ids": encoding.type_ids,
            }
        )
    if len(out) < 2 or len({len(row["input_ids"]) for row in out}) < 2:
        raise ValueError("corpus must exercise at least two differently sized rows for padded batch evidence")
    return out


def padded(rows: list[dict], token_type_ids: bool = True) -> dict[str, np.ndarray]:
    width = max(len(row["input_ids"]) for row in rows)
    keys = ("input_ids", "attention_mask", "token_type_ids") if token_type_ids else ("input_ids", "attention_mask")
    return {key: np.array([row[key] + [0] * (width - len(row[key])) for row in rows], dtype=np.int64) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("reference", "onnx"), required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1 through 64")
    candidate = json.loads(args.profile.read_text(encoding="utf-8"))
    validate_candidate_behavior(candidate)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    profile_hash = fixtures.sha256_file(args.profile)
    behavior = candidate["behavior"]
    uses_token_type_ids = CONTRACTS[candidate["origin"]["repo"]].get("token_type_ids", True)
    tokenizer = Tokenizer.from_file(str(args.snapshot / behavior["tokenizer_file"]))
    tokenizer.enable_truncation(behavior["max_tokens"], strategy="longest_first", direction="right")
    tokenizer.no_padding()
    if args.stage == "reference":
        if args.corpus is None:
            parser.error("reference requires --corpus")
        cases = load_cases(args.corpus, candidate, tokenizer)
    else:
        if args.reference is None:
            parser.error("onnx requires --reference")
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if reference["candidate_sha256"] != profile_hash or reference["profile"] != candidate["id"]:
            raise ValueError("reference identity differs from selected candidate")
        cases = [
            {key: row[key] for key in ("id", "text", "pair", "prompt", "input_ids", "attention_mask", "token_type_ids")}
            for row in reference["cases"]
        ]
    model, session = None, None
    if args.stage == "reference":
        if candidate["task"] == "embeddings":
            model = SentenceTransformer(str(args.snapshot), device="cpu", trust_remote_code=False)
            if model.max_seq_length != behavior["max_tokens"]:
                raise ValueError("sentence-transformers maximum sequence length differs from candidate")
        else:
            config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
            declared = config.get("sbert_ce_default_activation_function")
            extra = {}
            if CONTRACTS[candidate["origin"]["repo"]].get("score_activation_source", "config") == "config":
                if declared != "torch.nn.modules.linear.Identity":
                    raise ValueError("pinned reranker config must declare identity score activation")
            elif declared is not None or "sentence_transformers" in config:
                raise ValueError("contract passes identity explicitly; pinned config must not declare an activation")
            else:
                # CrossEncoder would default to sigmoid for one label; the upstream scores are raw logits.
                extra["activation_fn"] = torch.nn.Identity()
            model = CrossEncoder(
                str(args.snapshot), device="cpu", max_length=behavior["max_tokens"], trust_remote_code=False, **extra
            )
            activation = getattr(model, "activation_fn", getattr(model, "default_activation_function", None))
            if not isinstance(activation, torch.nn.Identity):
                raise ValueError("CrossEncoder loaded a non-identity activation")
        (model if candidate["task"] == "embeddings" else model.model).eval()
    else:
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(args.snapshot / behavior["graph_file"]), options, providers=["CPUExecutionProvider"]
        )
        expected_inputs = (
            {"input_ids", "attention_mask", "token_type_ids"}
            if uses_token_type_ids
            else {"input_ids", "attention_mask"}
        )
        if {value.name for value in session.get_inputs()} != expected_inputs:
            raise ValueError("unsupported ONNX graph input signature")

    def run(rows: list[dict]) -> tuple[np.ndarray, np.ndarray | None]:
        feeds = padded(rows, uses_token_type_ids)
        if session is not None:
            raw = session.run([behavior["output_name"]], feeds)[0]
            output = (
                pooled(raw, feeds["attention_mask"], behavior["pooling"], behavior["normalize"])
                if candidate["task"] == "embeddings"
                else raw
            )
            hidden = raw if candidate["task"] == "embeddings" else None
        else:
            tensors = {key: torch.from_numpy(value) for key, value in feeds.items()}
            with torch.inference_mode():
                if candidate["task"] == "embeddings":
                    result = model(tensors)
                    output, hidden = result["sentence_embedding"].numpy(), result["token_embeddings"].numpy()
                else:
                    output, hidden = model.model(**tensors).logits.numpy(), None
        expected_shape = (len(rows), candidate["behavior"]["hidden_size"] if candidate["task"] == "embeddings" else 1)
        if output.shape != expected_shape or not np.isfinite(output).all():
            raise ValueError(f"invalid output shape or non-finite values: {output.shape}, expected {expected_shape}")
        if hidden is not None and (
            hidden.shape != (len(rows), feeds["input_ids"].shape[1], candidate["behavior"]["hidden_size"])
            or not np.isfinite(hidden).all()
        ):
            raise ValueError("invalid hidden-state shape or non-finite values")
        return output.astype(np.float32), hidden

    layouts = []
    out_cases = []
    hidden_path = args.out / f"{args.stage}-hidden.f32"
    with hidden_path.open("wb") as tensor_file:
        for case in cases:
            output, hidden = run([case])
            row = copy.deepcopy(case)
            row["output"] = output[0].tolist()
            if candidate["task"] == "reranking":
                row["logits"], row["score"] = output[0].tolist(), float(output[0, 0])
            if args.stage == "reference" and candidate["task"] == "reranking":
                # An independent public CrossEncoder path catches activation/tokenization drift.
                score = model.predict(
                    [(case["text"], case["pair"])], batch_size=1, convert_to_numpy=True, show_progress_bar=False
                )
                row["original_score"] = float(np.asarray(score).reshape(-1)[0])
            if hidden is not None:
                values = hidden[0].astype("<f4")
                layouts.append({"id": case["id"], "offset": tensor_file.tell() // 4, "shape": list(values.shape)})
                tensor_file.write(values.tobytes())
            out_cases.append(row)
    if not layouts:
        hidden_path.unlink()
    batch_output = []
    chunks = []
    for start in range(0, len(cases), 8):
        chunk = cases[start : start + 8]
        values, _ = run(chunk)
        batch_output.extend(values.tolist())
        chunks.append(
            {
                "case_ids": [row["id"] for row in chunk],
                "input_lengths": [len(row["input_ids"]) for row in chunk],
                "width": max(len(row["input_ids"]) for row in chunk),
            }
        )
    document = {
        "format": "starlings.bert-candidate-results",
        "format_version": 1,
        "profile": candidate["id"],
        "task": candidate["task"],
        "stage": args.stage,
        "candidate_sha256": profile_hash,
        "reference_status": "proposed",
        "assets": [
            {"path": name, "sha256": fixtures.sha256_file(args.snapshot / name)}
            for name in (behavior["graph_file"], behavior["tokenizer_file"])
        ],
        "environment": {
            **fixtures.environment(
                ["torch", "transformers", "sentence-transformers", "tokenizers", "onnxruntime", "numpy"]
            ),
            "machine": __import__("platform").node(),
            "provider": "CPUExecutionProvider",
            "threads": args.threads,
            "onnxruntime": ort.__version__,
        },
        "cases": out_cases,
        "batches": [
            {
                "path": "ids",
                "case_ids": [row["id"] for row in cases],
                "input_lengths": [len(row["input_ids"]) for row in cases],
                "output": batch_output,
                "chunks": chunks,
            }
        ],
    }
    if args.reference is not None:
        document["reference_sha256"] = fixtures.sha256_file(args.reference)
    if layouts:
        document["hidden"] = {"file": hidden_path.name, "sha256": fixtures.sha256_file(hidden_path), "cases": layouts}
    fixtures.write_json(args.out / f"{args.stage}.json", document)
    print(f"{candidate['id']}: {args.stage}, {len(cases)} cases, original references remain proposed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
