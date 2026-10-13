"""Kev-0.8B: localize the ONNX Runtime against PyTorch difference layer by layer (diagnosis only, P-027).

Three steps, each in the environment that has the runtime it needs:

  row    (Kev environment) Write one question row (token ids, positions, readout offsets) to an .npz, from a step 2
         fixture (--fixture, --question) or from a suite record (--suite, --record-id, --question).
  ort    (onnxruntime environment) Copy the graph with every layer's intermediate tensors added as outputs (same
         external weights file), run the row and save: the residual stream entering each layer, the residual after
         each token mixer, each mixer output (DeltaNet out_proj or attention o_proj), each MLP output, and the final
         hidden states.
  torch  (Kev environment, --device cpu or cuda) Run the merged fp32 checkpoint (kev_merge.py) on the same row and
         compare with the ORT tensors in two ways:
           accumulated  torch's own residual stream against ORT's at each layer boundary (what the readout sees);
           local        each torch layer applied to ORT's input of that layer, against ORT's output of that layer,
                        split into the token mixer and the MLP. This is the error each ORT layer adds by itself.
         Local outputs are saved too, so two torch runs (CPU, CUDA) can be compared on identical inputs (`pair`).
  pair   Compare the local outputs of two torch runs (e.g. CPU against CUDA) on the same ORT inputs: the spread of
         two fp32 PyTorch implementations, a floor for the ORT local errors.

Numbers are max abs differences over all positions and over the readout positions, with the reference magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = 24


def residual_in(i: int) -> str:
    return "/model/embed_tokens/Gather/output_0" if i == 0 else f"/model/layers.{i}/input_layernorm/output_3"


def names(kinds: list[str]) -> dict[str, str]:
    """Graph tensor names of the intermediates, keyed by a short label."""
    out = {}
    for i in range(LAYERS):
        mixer = "linear_attn/out_proj" if kinds[i] == "linear_attention" else "attn/o_proj"
        out[f"in.{i}"] = residual_in(i)
        out[f"mixer.{i}"] = f"/model/layers.{i}/{mixer}/MatMul/output_0"
        out[f"post_mixer.{i}"] = f"/model/layers.{i}/post_attention_layernorm/output_3"
        out[f"mlp.{i}"] = f"/model/layers.{i}/mlp/down_proj/MatMul/output_0"
        if kinds[i] == "linear_attention":  # inside the DeltaNet mixer: delta-rule output, then gated RMSNorm
            out[f"delta_rule.{i}"] = f"/model/layers.{i}/linear_attn/LinearAttention/output_0"
            out[f"gated_norm.{i}"] = f"/model/layers.{i}/linear_attn/GatedRMSNorm/output_0"
    out["hidden"] = "hidden_states"
    return out


def layer_kinds(merged: Path) -> list[str]:
    cfg = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    return (cfg.get("text_config") or cfg)["layer_types"]


def cmd_row(args) -> None:
    if args.fixture:
        with np.load(args.fixture) as z:
            s_ids, s_pos = z["state_ids"], z["state_pos"]
            q = args.question
            ids = np.concatenate([s_ids, z[f"q{q}_ids"]])
            pos = np.concatenate([s_pos, z[f"q{q}_pos"]])
            readout = len(s_ids) + z[f"q{q}_readout"]
            source = f"{args.fixture.name} q{q}"
    else:
        from kev.checkpoint import Checkpoint
        from kev.data import materialize
        from kev.model import encode, load_tokenizer, rows_of
        from kev.suite import CONTEXT, load_split, read_manifest

        meta = Checkpoint(str(args.checkpoint)).meta
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        ctx = read_manifest(args.suite).get("context", CONTEXT)
        rec = next(r for r in load_split(args.suite, "development") if r["_meta"]["id"] == args.record_id)
        enc = encode(tok, materialize(rec), max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
        s_ids, s_pos, rows = rows_of(enc)
        qn = list(rec["questions"]).index(args.question_id)
        r = rows[qn]
        ids, pos = np.array(s_ids + r["ids"]), np.array(s_pos + r["pos"])
        readout = len(s_ids) + np.array([r["decide"], *r["opts"]])
        source = f"{args.suite} {args.record_id} {args.question_id}"
    np.savez(args.out, ids=ids.astype(np.int64), pos=pos.astype(np.int64), readout=readout.astype(np.int64))
    print(
        json.dumps({"row": str(args.out), "tokens": int(len(ids)), "options": int(len(readout) - 1), "source": source})
    )


def cmd_ort(args) -> None:
    import onnx
    import onnxruntime as ort

    kinds = layer_kinds(args.merged)
    want = names(kinds)
    model = onnx.load(str(args.onnx), load_external_data=False)
    present = {o for n in model.graph.node for o in n.output}
    missing = [t for t in want.values() if t not in present and t != "hidden_states"]
    if missing:
        raise SystemExit(f"tensors not in the graph: {missing[:5]}")
    for tensor in want.values():
        if tensor != "hidden_states":
            model.graph.output.append(onnx.helper.make_tensor_value_info(tensor, onnx.TensorProto.FLOAT, None))
    debug = args.onnx.parent / (args.onnx.stem + "-debug.onnx")  # beside the original, so the weights file resolves
    onnx.save(model, str(debug))
    opts = ort.SessionOptions()
    if args.threads:
        opts.intra_op_num_threads = args.threads
    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        ort.preload_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(debug), opts, providers=providers)
    with np.load(args.row) as z:
        ids, pos = z["ids"], z["pos"]
    feed = {"input_ids": ids[None], "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
    feed["position_ids"] = np.broadcast_to(pos[None], (3, 1, len(ids))).copy()
    for i in session.get_inputs():
        if i.name.startswith("past"):
            shape = [
                1 if k == 0 else d if isinstance(d, int) else (256 if d == "kv_cache_dim" else 0)
                for k, d in enumerate(i.shape)
            ]
            feed[i.name] = np.zeros(shape, dtype=np.float32)
    outs = session.run(list(want.values()), feed)
    np.savez(args.out, **{label: o[0] for label, o in zip(want, outs, strict=True)}, kinds=np.array(kinds))
    debug.unlink()
    print(json.dumps({"saved": str(args.out), "tensors": len(want)}))


def diff(a: np.ndarray, b: np.ndarray, readout: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max": float(d.max()),
        "readout_max": float(d[readout].max()),
        "ref_absmax": float(np.abs(b).max()),
    }


def delta_rule_fp64(query, key, value, g, beta) -> np.ndarray:
    """The gated delta rule token by token in float64 (transformers' torch_recurrent_gated_delta_rule, without its
    fp32 casts): l2-normalized q and k, q / sqrt(d_k), S = S * exp(g); S += k (beta (v - S^T k))^T; o = S^T q.
    Inputs [B, S, H, d]; returns [S, H * d_v]."""
    import torch

    q, k, v, b, gg = (x[0].transpose(0, 1).to("cpu", torch.float64) for x in (query, key, value, beta, g))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q / q.shape[-1] ** 0.5
    heads, n, dk = k.shape
    state = torch.zeros(heads, dk, v.shape[-1], dtype=torch.float64)
    out = torch.zeros(heads, n, v.shape[-1], dtype=torch.float64)
    for t in range(n):
        state = state * gg[:, t].exp()[:, None, None]
        kv = (state * k[:, t, :, None]).sum(-2)
        delta = (v[:, t] - kv) * b[:, t, None]
        state = state + k[:, t, :, None] * delta[:, None, :]
        out[:, t] = (state * q[:, t, :, None]).sum(-2)
    return out.transpose(0, 1).reshape(n, -1).numpy()


def cmd_torch(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    torch.set_grad_enabled(False)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    attn = "eager" if args.device == "cpu" else "sdpa"
    lm = AutoModelForCausalLM.from_pretrained(str(args.merged), dtype=torch.float32, attn_implementation=attn).model
    lm = lm.to(args.device).eval()
    if args.device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    with np.load(args.row) as z:
        ids, pos, readout = z["ids"], z["pos"], z["readout"]
    o = dict(np.load(args.ort_out))
    kinds = [str(k) for k in o.pop("kinds")]
    layer_kwargs, torch_out = {}, {}

    def pre_hook(i):
        def hook(module, a, kw):
            layer_kwargs[i] = {k: v for k, v in kw.items() if k != "past_key_values"}
            torch_out[f"in.{i}"] = (a[0] if a else kw["hidden_states"])[0].float().cpu().numpy()

        return hook

    handles = [layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True) for i, layer in enumerate(lm.layers)]
    dev = args.device
    h = lm(
        input_ids=torch.from_numpy(ids)[None].to(dev), position_ids=torch.from_numpy(pos)[None].to(dev)
    ).last_hidden_state
    for hd in handles:
        hd.remove()
    torch_out["hidden"] = h[0].float().cpu().numpy()

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x))[None].to(dev)

    report = {"device": dev, "tokens": int(len(ids)), "options": int(len(readout) - 1), "layers": []}
    local, exact = {}, {}
    for i, layer in enumerate(lm.layers):
        kw = layer_kwargs[i]
        x = t(o[f"in.{i}"])
        normed = layer.input_layernorm(x)
        if kinds[i] == "linear_attention":
            seen = {}

            def norm_hook(module, a, result, seen=seen):
                seen["delta_rule"], seen["gated_norm"] = a[0], result

            hd = layer.linear_attn.norm.register_forward_hook(norm_hook)
            rule = modeling.torch_chunk_gated_delta_rule

            def recording_rule(query, key, value, g, beta, rule=rule, seen=seen, **kw2):
                seen["rule_inputs"] = (query, key, value, g, beta)
                return rule(query, key, value, g=g, beta=beta, **kw2)

            modeling.torch_chunk_gated_delta_rule = recording_rule
            try:
                mixed = layer.linear_attn(
                    hidden_states=normed, cache_params=None, attention_mask=kw.get("attention_mask")
                )
            finally:
                modeling.torch_chunk_gated_delta_rule = rule
                hd.remove()
            for name in ("delta_rule", "gated_norm"):
                local[f"{name}.{i}"] = seen[name].reshape(len(ids), -1).float().cpu().numpy()
            exact[i] = delta_rule_fp64(*seen["rule_inputs"]).reshape(len(ids), -1)
        else:
            mixed, _ = layer.self_attn(
                hidden_states=normed,
                attention_mask=kw.get("attention_mask"),
                position_ids=kw.get("position_ids"),
                past_key_values=None,
                position_embeddings=kw["position_embeddings"],
            )
        post = t(o[f"post_mixer.{i}"])
        mlp = layer.mlp(layer.post_attention_layernorm(post))
        local[f"mixer.{i}"] = mixed[0].float().cpu().numpy()
        local[f"mlp.{i}"] = mlp[0].float().cpu().numpy()
        entry = {"layer": i, "kind": kinds[i]}
        entry["accumulated_residual_in"] = diff(o[f"in.{i}"], torch_out[f"in.{i}"], readout)
        entry["local_mixer"] = diff(o[f"mixer.{i}"], local[f"mixer.{i}"], readout)
        entry["local_mlp"] = diff(o[f"mlp.{i}"], local[f"mlp.{i}"], readout)
        if kinds[i] == "linear_attention":
            for name in ("delta_rule", "gated_norm"):
                entry[f"local_{name}"] = diff(o[f"{name}.{i}"], local[f"{name}.{i}"], readout)
            # fp64 recurrence on torch's delta-rule inputs (computed from ORT's layer input): the error of each
            # fp32 implementation against exact arithmetic. ORT's own inputs differ from these by projection and
            # convolution rounding (about 1e-6 relative), which is included in its number.
            entry["delta_rule_torch_vs_fp64"] = diff(local[f"delta_rule.{i}"], exact[i], readout)
            entry["delta_rule_ort_vs_fp64"] = diff(o[f"delta_rule.{i}"], exact[i], readout)
        report["layers"].append(entry)
    final = lm.norm(t(o[f"post_mixer.{LAYERS - 1}"]) + t(o[f"mlp.{LAYERS - 1}"]))
    local["hidden"] = final[0].float().cpu().numpy()
    report["final_local"] = diff(o["hidden"], local["hidden"], readout)
    report["final_accumulated"] = diff(o["hidden"], torch_out["hidden"], readout)
    np.savez(args.out.with_suffix(".npz"), **local)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for e in report["layers"]:
        print(
            f"{e['layer']:2d} {e['kind'][:6]:6s} acc_in {e['accumulated_residual_in']['readout_max']:.3e}"
            f" local_mixer {e['local_mixer']['max']:.3e} (ref {e['local_mixer']['ref_absmax']:.2f})"
            f" local_mlp {e['local_mlp']['max']:.3e} (ref {e['local_mlp']['ref_absmax']:.2f})"
            + (
                f" delta_rule {e['local_delta_rule']['max']:.3e} (ref {e['local_delta_rule']['ref_absmax']:.2f})"
                f" vs fp64: torch {e['delta_rule_torch_vs_fp64']['max']:.3e}"
                f" ort {e['delta_rule_ort_vs_fp64']['max']:.3e}"
                if "local_delta_rule" in e
                else ""
            )
        )
    print("final local", report["final_local"], "final accumulated", report["final_accumulated"])


def cmd_pair(args) -> None:
    a, b = dict(np.load(args.a)), dict(np.load(args.b))
    with np.load(args.row) as z:
        readout = z["readout"]
    rows = []
    for k in a:
        rows.append({"tensor": k, **diff(a[k], b[k], readout)})
    args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['tensor']:10s} {r['max']:.3e} readout {r['readout_max']:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("row")
    p.add_argument("--fixture", type=Path)
    p.add_argument("--question", type=int, default=0)
    p.add_argument("--suite", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--record-id")
    p.add_argument("--question-id")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("ort")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--row", type=Path, required=True)
    p.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("torch")
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--row", type=Path, required=True)
    p.add_argument("--ort-out", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pair")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--row", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"row": cmd_row, "ort": cmd_ort, "torch": cmd_torch, "pair": cmd_pair}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
