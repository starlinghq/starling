"""Kev-0.8B spike, steps 3 (check) and 5: compare the exported backbone with the PyTorch reference.

Subcommands:
  merged    Load the merged fp32 checkpoint (kev_merge.py) as a transformers text model and recompute the step 2
            fixture probabilities (forward layout). Pass bar from kev.md step 3: max abs difference 1e-6.
  fixtures  Python ONNX Runtime (CPU EP) on the step 2 fixtures: (1) single rows without cache, hidden states at
            readout indices and probabilities; (2) state prefix then question rows from the carried state; (3) right-
            padded batches of rows against each row alone.
  suite     Full development partition through ONNX Runtime in the forward layout, against kev.benchmark's
            predictions.jsonl from the same checkpoint: probabilities, argmax flips, near ties, accuracy.
  long      The long-state fixtures (about 2k and 8k state tokens, or the --long-tokens fixture of kev_fixtures.py):
            probabilities, latency, peak RSS and, on CUDA, peak device memory. Optionally one synthetic state of
            random token ids for memory and time only.
  placement Which execution provider each graph node ran on (session profiling). Plan 11 requires fallback to be
            visible; --log-severity 0 also prints ONNX Runtime's own node-assignment log.
  compare   Two prediction files (kev.benchmark output directory, or this script's suite -rows.jsonl) against each
            other: probabilities, argmax flips with the reference top-two margin, accuracy.

--provider cpu|cuda selects the execution provider (cuda: CUDAExecutionProvider, CPU for nodes CUDA does not take).
fp16 and bf16 graphs are handled: inputs go in the graph's dtype and hidden states come back as fp32 (bf16 through
OrtValue, which numpy cannot represent).

Every result also reports a NaN/Inf scan. Thresholds are the provisional ones in kev.md step 8; this script reports
numbers and does not decide pass or fail.

Usage (Kev environment with onnxruntime 1.30.0 or onnxruntime-gpu 1.30.0, from the Kev checkout):
    python kev_ort.py fixtures --onnx <dir>/model.onnx --head <merged dir> --fixtures <dir> --out <json>
    python kev_ort.py suite --provider cuda --onnx .. --head .. --suite evals/v7/decision-v7 --reference <run> --out ..
    python kev_ort.py compare --reference <run dir or rows.jsonl> --candidate <..> --suite <suite> --out <json>
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Head:
    def __init__(self, merged_dir: Path):
        w = load_file(str(merged_dir / "head.safetensors"))
        meta = json.loads((merged_dir / "head.json").read_text(encoding="utf-8"))
        self.qw, self.qb, self.kw, self.kb = w["q.weight"], w["q.bias"], w["k.weight"], w["k.bias"]
        self.scale = np.float32(1 / np.sqrt(meta["head_dim"]))
        self.temperature = np.float32(meta["temperature"])

    def logits(self, h_decide: np.ndarray, h_opts: np.ndarray) -> np.ndarray:
        q = h_decide @ self.qw.T + self.qb
        k = h_opts @ self.kw.T + self.kb
        return (k @ q) * self.scale / self.temperature

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()


BF16 = 16  # ONNX TensorProto.BFLOAT16


def to_float32(value) -> np.ndarray:
    """numpy array or CPU-resident OrtValue -> float32 numpy. bf16 has no numpy type, so its raw bits are widened."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if value.data_type() != "tensor(bfloat16)":
        return value.numpy().astype(np.float32, copy=False)
    n = int(np.prod(value.shape()))
    raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(value.data_ptr())).copy()
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(value.shape())


def providers_for(provider: str, device_id: int = 0, tf32: bool = False) -> list:
    """CUDA EP options: use_tf32 defaults to on in ONNX Runtime (TF32 MatMul on Ampere and later); off here unless
    asked for, to match the fp32-exact PyTorch reference (kev.predictors.LocalPredictor turns TF32 off too)."""
    if provider == "cuda":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # the CUDA 13 and cuDNN 9 wheels onnxruntime-gpu 1.30.0 names as extras
        options = {"device_id": device_id, "use_tf32": "1" if tf32 else "0"}
        return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class GpuMemory:
    """Peak device memory of this process, sampled from nvidia-smi (ONNX Runtime exposes no peak counter)."""

    def __init__(self, period: float = 0.1):
        self.period, self.peak_mb, self._stop = period, 0.0, threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> float:
        query = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        out = subprocess.run(query, capture_output=True, text=True, check=False).stdout
        rows = (line.split(", ") for line in out.splitlines() if line)
        return sum(float(mb) for pid, mb in rows if int(pid) == os.getpid())

    def _loop(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._sample())
            self._stop.wait(self.period)

    def start(self) -> GpuMemory:
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self.peak_mb


class Backbone:
    """Plain ONNX Runtime session over the GenAI builder graph (no generation loop)."""

    def __init__(
        self,
        path: Path,
        threads: int = 0,
        kv_dim: int = 256,
        provider: str = "cpu",
        tf32: bool = False,
        profile_prefix: str | None = None,
        log_severity: int | None = None,
    ):
        self.kv_dim = kv_dim  # attention head_dim; the builder leaves the KV cache's last dim symbolic
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        if profile_prefix:
            opts.enable_profiling = True
            opts.profile_file_prefix = profile_prefix
        if log_severity is not None:
            opts.log_severity_level = log_severity
        self.provider = provider
        self.session = ort.InferenceSession(str(path), opts, providers=providers_for(provider, tf32=tf32))
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.past_names = [n for n in self.inputs if n.startswith("past")]
        self.hidden_name = next(o for o in self.outputs if "hidden" in o)
        self.bf16 = any(i.type == "tensor(bfloat16)" for i in self.inputs.values())

    @staticmethod
    def _np_type(t: str):
        return {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bfloat16)": np.uint16,  # raw bits, fed as an OrtValue of ONNX type bfloat16
            "tensor(int64)": np.int64,
        }[t]

    def empty_past(self, batch: int) -> dict[str, np.ndarray]:
        past = {}
        for name in self.past_names:
            meta = self.inputs[name]
            shape = []
            for i, d in enumerate(meta.shape):
                if isinstance(d, int):
                    shape.append(d)
                elif i == 0:
                    shape.append(batch)
                elif d == "kv_cache_dim":
                    shape.append(self.kv_dim)
                else:
                    shape.append(0)  # past sequence length
            zeros = np.zeros(shape, dtype=self._np_type(meta.type))
            past[name] = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(zeros, BF16) if self.bf16 else zeros
        return past

    def run(self, ids: np.ndarray, pos: np.ndarray, mask: np.ndarray, past: dict | None = None):
        """ids, pos [B,S]; mask [B, past+S]. Returns (hidden [B,S,d] as fp32, present dict keyed by matching past
        name). Present values are numpy arrays, or OrtValues for a bf16 graph; feed them back unchanged."""
        batch = ids.shape[0]
        feed = {"input_ids": ids.astype(np.int64), "attention_mask": mask.astype(np.int64)}
        if "position_ids" in self.inputs:
            rank = len(self.inputs["position_ids"].shape)
            feed["position_ids"] = np.broadcast_to(pos, (3, *pos.shape)).copy() if rank == 3 else pos
        feed.update(past if past is not None else self.empty_past(batch))
        if self.bf16:
            feed = {
                k: v if isinstance(v, ort.OrtValue) else ort.OrtValue.ortvalue_from_numpy(v) for k, v in feed.items()
            }
            out = self.session.run_with_ort_values(self.outputs, feed)
        else:
            out = self.session.run(None, feed)
        named = dict(zip(self.outputs, out, strict=True))
        present = {}
        for name in self.past_names:
            pres = name.replace("past_key_values", "present").replace("past", "present", 1)
            present[name] = named[pres]
        return to_float32(named[self.hidden_name]), present


def clone(past: dict) -> dict:
    """A copy of a carried state for one branch. OrtValues are never written by a run, so they are shared."""
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in past.items()}


def make_backbone(args, **kw) -> Backbone:
    return Backbone(args.onnx, args.threads, provider=args.provider, tf32=args.cuda_tf32, **kw)


def load_fixture(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def questions(fx: dict) -> int:
    return sum(1 for k in fx if k.endswith("_readout"))


def stats(values: list[float] | np.ndarray) -> dict:
    a = np.asarray(values, dtype=np.float64).ravel()
    if not a.size:
        return {"n": 0}
    return {
        "n": int(a.size),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "p50": float(np.quantile(a, 0.5)),
        "p99": float(np.quantile(a, 0.99)),
    }


def finite(*arrays) -> bool:
    return all(np.isfinite(a).all() for a in arrays)


def cmd_fixtures(args) -> dict:
    bb, head = make_backbone(args), Head(args.head)
    idx = json.loads((args.fixtures / "index.json").read_text())
    single_h, single_p, cached_p, cached_vs_single, batch_vs_single, flips = [], [], [], [], [], []
    nonfinite = []
    per_record = []
    csharp_cases = []
    for entry in idx:
        fx = load_fixture(args.fixtures / entry["file"])
        s_ids, s_pos = fx["state_ids"], fx["state_pos"]
        rows = []
        rec_p = []
        for q in range(questions(fx)):
            ids = np.concatenate([s_ids, fx[f"q{q}_ids"]])
            pos = np.concatenate([s_pos, fx[f"q{q}_pos"]])
            readout = len(s_ids) + fx[f"q{q}_readout"]
            hidden, _ = bb.run(ids[None], pos[None], np.ones((1, len(ids))))
            sel = hidden[0, readout]
            p = head.softmax(head.logits(sel[0], sel[1:]))
            ref_h, ref_p = fx[f"q{q}_forward_hidden"], fx[f"q{q}_forward_probs"]
            single_h.append(np.abs(sel - ref_h).max())
            single_p.append(np.abs(p - ref_p).max())
            flips.append(int(np.argmax(p) != np.argmax(ref_p)))
            if not finite(hidden, p):
                nonfinite.append(entry["id"])
            rows.append((ids, pos, readout, p))
            rec_p.append(float(np.abs(p - ref_p).max()))
            if len(csharp_cases) < args.export_csharp:
                csharp_cases.append(
                    {
                        "id": entry["id"],
                        "ids": ids.tolist(),
                        "pos": pos.tolist(),
                        "readout": readout.tolist(),
                        "ort_python_probs": p.tolist(),
                        "ort_python_hidden": sel.tolist(),
                    }
                )
        # (2) state once, then each branch from a copy of the carried state.
        hidden_s, present = bb.run(s_ids[None], s_pos[None], np.ones((1, len(s_ids))))
        for q in range(questions(fx)):
            b_ids, b_pos = fx[f"q{q}_ids"], fx[f"q{q}_pos"]
            mask = np.ones((1, len(s_ids) + len(b_ids)))
            hidden, _ = bb.run(b_ids[None], b_pos[None], mask, clone(present))
            sel = hidden[0, fx[f"q{q}_readout"]]
            p = head.softmax(head.logits(sel[0], sel[1:]))
            cached_p.append(np.abs(p - fx[f"q{q}_probs_probs"]).max())
            cached_vs_single.append(np.abs(p - rows[q][3]).max())
            if not finite(hidden, hidden_s, p):
                nonfinite.append(entry["id"])
        # (3) all rows of the record as one right-padded batch.
        if len(rows) > 1 or args.batch_with_neighbours:
            width = max(len(r[0]) for r in rows)
            ids_b = np.zeros((len(rows), width), dtype=np.int64)
            pos_b = np.zeros((len(rows), width), dtype=np.int64)
            mask_b = np.zeros((len(rows), width), dtype=np.int64)
            for i, (ids, pos, _, _) in enumerate(rows):
                ids_b[i, : len(ids)], pos_b[i, : len(ids)], mask_b[i, : len(ids)] = ids, pos, 1
            ids_b[mask_b == 0] = args.pad_id
            hidden, _ = bb.run(ids_b, pos_b, mask_b)
            for i, (_, _, readout, p_alone) in enumerate(rows):
                sel = hidden[i, readout]
                p = head.softmax(head.logits(sel[0], sel[1:]))
                batch_vs_single.append(np.abs(p - p_alone).max())
        per_record.append({"id": entry["id"], "questions": len(rows), "max_abs_prob": max(rec_p)})
    if csharp_cases:
        (args.out.parent / "csharp-cases.json").write_text(json.dumps(csharp_cases), encoding="utf-8")
    return {
        "records": len(idx),
        "questions": len(single_p),
        "1_single_rows": {
            "hidden_at_readouts_max_abs": stats(single_h),
            "probability_max_abs": stats(single_p),
            "argmax_flips": int(sum(flips)),
        },
        "2_cached_state_rows": {
            "probability_vs_torch_probs_layout": stats(cached_p),
            "probability_vs_ort_single_rows": stats(cached_vs_single),
        },
        "3_padded_batches": {
            "probability_vs_row_alone": stats(batch_vs_single),
            "note": "batches are a record's question rows (different lengths); single-question records skipped",
        },
        "nonfinite_records": sorted(set(nonfinite)),
        "per_record": per_record,
    }


def cmd_suite(args) -> dict:
    import torch
    from kev.api import question_keys
    from kev.checkpoint import Checkpoint
    from kev.data import materialize
    from kev.model import load_tokenizer, rows_of
    from kev.suite import CONTEXT, load_split, read_manifest

    torch.set_grad_enabled(False)
    meta = Checkpoint(str(args.checkpoint)).meta
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    from kev.model import encode

    bb, head = make_backbone(args), Head(args.head)
    ctx = read_manifest(args.suite).get("context", CONTEXT)
    records = load_split(args.suite, "development")
    reference = {}
    for line in (args.reference / "predictions.jsonl").read_text(encoding="utf-8").split("\n"):
        if line:
            row = json.loads(line)
            reference[row["id"]] = row["prediction"]
    diffs, flips, near_ties, nonfinite, latency = [], [], [], [], []
    correct_ref = correct_ort = total = 0
    out_rows = []
    t0 = time.perf_counter()
    for n, rec in enumerate(records):
        if args.limit and n >= args.limit:
            break
        rid = rec["_meta"]["id"]
        ref = reference.get(rid)
        if ref is None:
            continue
        enc = encode(tok, materialize(rec), max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
        s_ids, s_pos, rows = rows_of(enc)
        for (qid, q), r in zip(rec["questions"].items(), rows, strict=True):
            ids = np.array(s_ids + r["ids"])
            pos = np.array(s_pos + r["pos"])
            readout = len(s_ids) + np.array([r["decide"], *r["opts"]])
            t_q = time.perf_counter()
            hidden, _ = bb.run(ids[None], pos[None], np.ones((1, len(ids))))
            sel = hidden[0, readout]
            p = head.softmax(head.logits(sel[0], sel[1:]))
            latency.append((len(ids), time.perf_counter() - t_q))
            keys = question_keys(q["type"], q.get("criteria"))
            ref_p = np.array([ref["probabilities"][qid][k] for k in keys])
            d = float(np.abs(p - ref_p).max())
            diffs.append(d)
            top2 = np.sort(ref_p)[-2:]
            margin = float(top2[1] - top2[0]) if len(ref_p) > 1 else 1.0
            flip = int(np.argmax(p) != np.argmax(ref_p))
            flips.append(flip)
            if flip:
                near_ties.append({"id": rid, "question": qid, "reference_margin": margin, "max_abs": d})
            if not finite(hidden, p):
                nonfinite.append(rid)
            label = keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])
            if rec["_meta"].get("variant", "clean") == "clean" and rec["_meta"].get("source") != "unknowable":
                total += 1
                correct_ref += int(np.argmax(ref_p) == label)
                correct_ort += int(np.argmax(p) == label)
            out_rows.append({"id": rid, "question": qid, "p": p.tolist()})
        if n % 100 == 0:
            print(n, f"{time.perf_counter() - t0:.0f}s", f"max={max(diffs):.3g}", flush=True)
    (args.out.parent / (args.out.stem + "-rows.jsonl")).write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n", encoding="utf-8"
    )
    return {
        "suite": str(args.suite),
        "questions": len(diffs),
        "probability_max_abs": stats(diffs),
        "argmax_flips": int(sum(flips)),
        "flips": near_ties,
        "accuracy_clean_knowable": {
            "n": total,
            "reference": correct_ref / total if total else None,
            "onnxruntime": correct_ort / total if total else None,
        },
        "nonfinite_records": sorted(set(nonfinite)),
        "seconds": time.perf_counter() - t0,
        "peak_rss_mb": rss_mb(),
        "provider": args.provider,
        "cuda_tf32": args.cuda_tf32,
        "threads": args.threads,
        "latency_per_question_seconds": latency_stats(latency),
    }


def latency_stats(latency: list[tuple[int, float]]) -> dict:
    """Wall time per question of one ORT row (state + branch, forward layout, host-device copies included). The
    first question is the warm-up and is left out."""
    warm = latency[1:]
    if not warm:
        return {"n": 0}
    secs = np.array([t for _, t in warm])
    toks = np.array([n for n, _ in warm])
    return {
        "n": len(warm),
        "mean": float(secs.mean()),
        "p50": float(np.quantile(secs, 0.5)),
        "p90": float(np.quantile(secs, 0.9)),
        "tokens_mean": float(toks.mean()),
        "questions_per_second": float(len(warm) / secs.sum()),
        "tokens_per_second": float(toks.sum() / secs.sum()),
    }


def cmd_merged(args) -> dict:
    """Two checks on the merged checkpoint. (a) Its language-model tensors against Kev's own in-memory fp32 merge,
    bit for bit. (b) The step 2 probabilities recomputed from it, both with each question row alone and with the
    rows of a record batched and right-padded the way kev.model.DecisionModel._rows_hidden runs them."""
    import os

    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    torch.set_grad_enabled(False)
    result = {}
    if args.checkpoint:
        if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") != "1":
            raise SystemExit("set TORCH_FORCE_WEIGHTS_ONLY_LOAD=1")
        _, kev_model = Checkpoint(str(args.checkpoint)).load("cpu", LoadOptions())
        ref = kev_model.lm.state_dict()
        unequal, compared = [], 0
        with safe_open(str(args.head / "model.safetensors"), framework="pt") as f:
            for key, value in ref.items():
                saved = f.get_tensor("model.language_model." + key)
                compared += 1
                if not torch.equal(saved, value.float()):
                    unequal.append(key)
        result["a_tensors_bit_identical_to_kev_merge"] = {"compared": compared, "unequal": unequal}
        del kev_model, ref
    lm = AutoModelForCausalLM.from_pretrained(str(args.head), dtype=torch.float32, attn_implementation="eager").model
    lm.eval()
    head = Head(args.head)
    idx = json.loads((args.fixtures / "index.json").read_text())
    alone, batched = [], []
    for entry in idx[: args.limit or None]:
        fx = load_fixture(args.fixtures / entry["file"])
        rows = []
        for q in range(questions(fx)):
            ids = np.concatenate([fx["state_ids"], fx[f"q{q}_ids"]])
            pos = np.concatenate([fx["state_pos"], fx[f"q{q}_pos"]])
            readout = len(fx["state_ids"]) + fx[f"q{q}_readout"]
            rows.append((ids, pos, readout, fx[f"q{q}_forward_probs"]))
            h = lm(input_ids=torch.from_numpy(ids)[None], position_ids=torch.from_numpy(pos)[None]).last_hidden_state
            sel = h[0, readout].float().numpy()
            alone.append(float(np.abs(head.softmax(head.logits(sel[0], sel[1:])) - rows[-1][3]).max()))
        width = max(len(r[0]) for r in rows)
        ids_b = torch.full((len(rows), width), args.pad_id, dtype=torch.long)
        pos_b = torch.zeros((len(rows), width), dtype=torch.long)
        att_b = torch.zeros((len(rows), width), dtype=torch.long)
        for i, (ids, pos, _, _) in enumerate(rows):
            ids_b[i, : len(ids)], pos_b[i, : len(ids)], att_b[i, : len(ids)] = (
                torch.from_numpy(ids),
                torch.from_numpy(pos),
                1,
            )
        h = lm(input_ids=ids_b, position_ids=pos_b, attention_mask=att_b).last_hidden_state
        for i, (_, _, readout, ref_p) in enumerate(rows):
            sel = h[i, readout].float().numpy()
            batched.append(float(np.abs(head.softmax(head.logits(sel[0], sel[1:])) - ref_p).max()))
    result["b_rows_alone_vs_step2"] = stats(alone)
    result["b_rows_batched_like_kev_vs_step2"] = stats(batched)
    result["questions"] = len(alone)
    return result


def run_state(bb: Backbone, ids: np.ndarray, pos: np.ndarray, chunk: int = 0):
    """The state in one pass, or (chunk > 0) in pieces of `chunk` tokens, each continuing from the carried cache.
    Returns the last piece's hidden states and the final present state."""
    if not chunk:
        return bb.run(ids[None], pos[None], np.ones((1, len(ids))))
    present = None
    for start in range(0, len(ids), chunk):
        end = min(start + chunk, len(ids))
        hidden, present = bb.run(ids[None, start:end], pos[None, start:end], np.ones((1, end)), present)
    return hidden, present


def cmd_long(args) -> dict:
    bb, head = make_backbone(args), Head(args.head)
    gpu = GpuMemory().start() if args.provider == "cuda" else None
    results = []
    for path in sorted(args.fixtures.glob("documents-v1-*.npz")) if args.fixtures else []:
        fx = load_fixture(path)
        s_ids, s_pos = fx["state_ids"], fx["state_pos"]
        t0 = time.perf_counter()
        hidden_s, present = run_state(bb, s_ids, s_pos, args.state_chunk)
        t_state = time.perf_counter() - t0
        res = {"file": path.name, "state_tokens": int(len(s_ids)), "state_seconds": t_state, "questions": []}
        res["state_chunk"] = args.state_chunk or None
        print(json.dumps({"file": path.name, "state_seconds": t_state}), flush=True)
        for q in range(questions(fx)):
            ids = np.concatenate([s_ids, fx[f"q{q}_ids"]])
            pos = np.concatenate([s_pos, fx[f"q{q}_pos"]])
            t1 = time.perf_counter()
            if args.state_chunk:  # no single pass over state + branch; the forward layout is not run
                hidden, p, t_row = None, None, None
            else:
                hidden, _ = bb.run(ids[None], pos[None], np.ones((1, len(ids))))
                t_row = time.perf_counter() - t1
                sel = hidden[0, len(s_ids) + fx[f"q{q}_readout"]]
                p = head.softmax(head.logits(sel[0], sel[1:]))
            b_ids, b_pos = fx[f"q{q}_ids"], fx[f"q{q}_pos"]
            t2 = time.perf_counter()
            hidden_c, _ = bb.run(b_ids[None], b_pos[None], np.ones((1, len(ids))), clone(present))
            t_branch = time.perf_counter() - t2
            sel_c = hidden_c[0, fx[f"q{q}_readout"]]
            p_c = head.softmax(head.logits(sel_c[0], sel_c[1:]))
            res["questions"].append(
                {
                    "row_seconds": t_row,
                    "branch_from_state_seconds": t_branch,
                    "branch_tokens": int(len(b_ids)),
                    "forward_probability_max_abs": None
                    if p is None
                    else float(np.abs(p - fx[f"q{q}_forward_probs"]).max()),
                    "cached_probability_max_abs": float(np.abs(p_c - fx[f"q{q}_probs_probs"]).max()),
                    "argmax_flip": int(np.argmax(p if p is not None else p_c) != np.argmax(fx[f"q{q}_forward_probs"])),
                    "finite": bool(finite(*(a for a in (hidden, hidden_s, hidden_c, p, p_c) if a is not None))),
                }
            )
        res["peak_rss_mb"] = rss_mb()
        if gpu:
            res["gpu_peak_mb_so_far"] = gpu.peak_mb
        results.append(res)
        print(json.dumps(res), flush=True)
    for n in [int(x) for x in args.random_states.split(",") if x]:
        # memory and time only: one state of random token ids per length, smallest first, stop at the first failure
        ids = np.random.default_rng(0).integers(1000, 200000, size=n)
        t0 = time.perf_counter()
        try:
            hidden, _ = bb.run(ids[None], np.arange(n)[None], np.ones((1, n)))
        except Exception as error:  # noqa: BLE001 - an out-of-memory failure is the measurement
            results.append({"synthetic_state_tokens": n, "error": str(error)[:2000], "peak_rss_mb": rss_mb()})
            print(json.dumps(results[-1]), flush=True)
            break
        results.append(
            {
                "synthetic_state_tokens": n,
                "note": "random token ids, memory and time only",
                "seconds": time.perf_counter() - t0,
                "peak_rss_mb": rss_mb(),
                "gpu_peak_mb_so_far": gpu.peak_mb if gpu else None,
                "finite": bool(finite(hidden)),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    return {"long_states": results, "provider": args.provider, "gpu_peak_mb": gpu.stop() if gpu else None}


def cmd_placement(args) -> dict:
    """One fixture row with session profiling; nodes counted per execution provider. Profile Node events carry the
    provider that ran each kernel, so nodes the preferred provider did not take are listed by op type."""
    prefix = str(args.out.parent / (args.out.stem + "-profile"))
    bb = make_backbone(args, profile_prefix=prefix, log_severity=args.log_severity)
    fx = load_fixture(sorted(args.fixtures.glob("r*.npz"))[0])
    ids = np.concatenate([fx["state_ids"], fx["q0_ids"]])
    pos = np.concatenate([fx["state_pos"], fx["q0_pos"]])
    for _ in range(2):
        hidden, _ = bb.run(ids[None], pos[None], np.ones((1, len(ids))))
    profile = bb.session.end_profiling()
    nodes = {}
    for ev in json.loads(Path(profile).read_text(encoding="utf-8")):
        a = ev.get("args", {})
        if ev.get("cat") == "Node" and "provider" in a and ev["name"].endswith("_kernel_time"):
            nodes[ev["name"][: -len("_kernel_time")]] = (a.get("op_name"), a["provider"])
    preferred = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    return {
        "profile_file": profile,
        "session_providers": bb.session.get_providers(),
        "nodes_profiled": len(nodes),
        "nodes_by_provider": dict(Counter(p for _, p in nodes.values())),
        "nodes_not_on_preferred_provider_by_op": dict(
            Counter(f"{op} -> {p}" for op, p in nodes.values() if p != preferred)
        ),
        "finite": bool(finite(hidden)),
        "row_tokens": int(len(ids)),
    }


def read_predictions(path: Path) -> dict:
    """kev.benchmark output directory (probabilities keyed by option name) or this script's suite -rows.jsonl
    (probabilities in option order), keyed by (record id, question id)."""
    if path.is_dir():
        path = path / "predictions.jsonl"
    rows = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if "prediction" in row:
            for qid, probs in row["prediction"]["probabilities"].items():
                rows[(row["id"], qid)] = probs
        else:
            rows[(row["id"], row["question"])] = np.array(row["p"])
    return rows


def cmd_compare(args) -> dict:
    from kev.api import question_keys
    from kev.suite import load_split

    ref, cand = read_predictions(args.reference), read_predictions(args.candidate)
    diffs, flips, nonfinite, total = [], [], 0, 0
    correct = {"reference": 0, "candidate": 0}
    for rec in load_split(args.suite, "development"):
        rid = rec["_meta"]["id"]
        for qid, q in rec["questions"].items():
            if (rid, qid) not in ref or (rid, qid) not in cand:
                continue
            keys = question_keys(q["type"], q.get("criteria"))
            r, c = (
                np.array([v[k] for k in keys]) if isinstance(v, dict) else v for v in (ref[rid, qid], cand[rid, qid])
            )
            nonfinite += int(not (np.isfinite(r).all() and np.isfinite(c).all()))
            d = float(np.abs(r - c).max())
            diffs.append(d)
            if np.argmax(r) != np.argmax(c):
                top2 = np.sort(r)[-2:]
                flips.append({"id": rid, "question": qid, "reference_margin": float(top2[1] - top2[0]), "max_abs": d})
            if rec["_meta"].get("variant", "clean") == "clean" and rec["_meta"].get("source") != "unknowable":
                label = keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])
                total += 1
                correct["reference"] += int(np.argmax(r) == label)
                correct["candidate"] += int(np.argmax(c) == label)
    return {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "questions_compared": len(diffs),
        "reference_questions": len(ref),
        "candidate_questions": len(cand),
        "probability_max_abs": stats(diffs),
        "argmax_flips": len(flips),
        "flips": flips,
        "nonfinite_questions": nonfinite,
        "accuracy_clean_knowable": {"n": total, **{k: v / total if total else None for k, v in correct.items()}},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fixtures", "suite", "long", "merged", "placement", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--onnx", type=Path)
        p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--cuda-tf32", action="store_true", help="allow TF32 MatMul on the CUDA EP (ORT's default)")
        p.add_argument("--log-severity", type=int, default=None, help="ONNX Runtime session log level (0 = verbose)")
        p.add_argument("--candidate", type=Path, help="compare: benchmark directory or -rows.jsonl to compare")
        p.add_argument("--random-states", default="", help="long: comma list of random-id state lengths")
        p.add_argument("--state-chunk", type=int, default=0, help="long: run the state in pieces of this many tokens")
        p.add_argument("--head", type=Path, help="merged directory with head.safetensors and head.json")
        p.add_argument("--fixtures", type=Path)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--threads", type=int, default=0)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--pad-id", type=int, default=248044)
        p.add_argument("--batch-with-neighbours", action="store_true")
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--suite", type=Path)
        p.add_argument("--reference", type=Path, help="kev.benchmark output directory")
        p.add_argument("--export-csharp", type=int, default=0, help="write this many single-row cases for the C# smoke")
    args = parser.parse_args()
    commands = {
        "fixtures": cmd_fixtures,
        "suite": cmd_suite,
        "long": cmd_long,
        "merged": cmd_merged,
        "placement": cmd_placement,
        "compare": cmd_compare,
    }
    result = commands[args.cmd](args)
    result["onnxruntime"] = ort.__version__
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("per_record", "flips", "long_states")}
    print(json.dumps(summary, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
