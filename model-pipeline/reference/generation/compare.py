"""Compare a candidate backend's greedy results with the transformers reference.

Shared result format (one JSON object per line in ``results.jsonl``; written by ``run_reference.py`` and by the
C# runners):

- ``id`` (str), ``backend`` (str);
- ``prompt_ids`` (list[int]): the ids actually fed to the model. Candidates are fed the reference ids;
- ``output_ids`` (list[int]): generated ids, including the stop token if one was produced;
- ``finish_reason``: ``"eos"`` or ``"length"``;
- optional ``rendered`` (str) and ``backend_prompt_ids`` (list[int]): the prompt as the backend's own chat template
  and tokenizer produce it, compared with the reference rendering and ids;
- optional ``step_margin`` (list[float]): top-1 minus top-2 logit per step (reference only).

Step-0 log-softmax vectors are float32 little-endian files ``step0/<id>.f32`` next to ``results.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in (path / "results.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def step0(path: Path, pid: str) -> np.ndarray | None:
    f = path / "step0" / f"{pid}.f32"
    return np.fromfile(f, dtype="<f4") if f.exists() else None


def compare(ref_dir: Path, cand_dir: Path) -> dict:
    ref, cand = load(ref_dir), load(cand_dir)
    rows = []
    for pid, r in ref.items():
        c = cand.get(pid)
        if c is None:
            rows.append({"id": pid, "missing": True})
            continue
        ro, co = r["output_ids"], c["output_ids"]
        n = common_prefix(ro, co)
        exact = ro == co
        row = {
            "id": pid,
            "prompt_ids_equal": c["prompt_ids"] == r["prompt_ids"],
            "prompt_len": len(r["prompt_ids"]),
            "ref_len": len(ro),
            "cand_len": len(co),
            "exact": exact,
            "match_len": n,
            "first_divergence": None if exact else n,
            "ref_margin_at_divergence": None,
            "finish_ref": r["finish_reason"],
            "finish_cand": c["finish_reason"],
        }
        if not exact and "step_margin" in r and n < len(r["step_margin"]):
            row["ref_margin_at_divergence"] = r["step_margin"][n]
        if "rendered" in c and c["rendered"] is not None:
            row["rendered_equal"] = c["rendered"] == r["rendered"]
        if "backend_prompt_ids" in c and c["backend_prompt_ids"] is not None:
            row["backend_prompt_ids_equal"] = c["backend_prompt_ids"] == r["prompt_ids"]
        a, b = step0(ref_dir, pid), step0(cand_dir, pid)
        if a is not None and b is not None and a.shape == b.shape:
            top = np.argsort(-a)[:5]
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            row["step0_max_abs_logprob"] = float(d.max())
            row["step0_top5_max_abs_logprob"] = float(d[top].max())
            row["step0_argmax_equal"] = int(a.argmax()) == int(b.argmax())
        rows.append(row)
    present = [x for x in rows if not x.get("missing")]
    summary = {
        "prompts": len(rows),
        "missing": len(rows) - len(present),
        "exact": sum(x["exact"] for x in present),
        "token_match_fraction": (sum(x["match_len"] for x in present) / max(1, sum(x["ref_len"] for x in present))),
        "finish_equal": sum(x["finish_ref"] == x["finish_cand"] for x in present),
        "step0_argmax_equal": sum(bool(x.get("step0_argmax_equal")) for x in present),
        "step0_max_abs_logprob": max(
            (x["step0_max_abs_logprob"] for x in present if "step0_max_abs_logprob" in x), default=None
        ),
        "step0_top5_max_abs_logprob": max(
            (x["step0_top5_max_abs_logprob"] for x in present if "step0_top5_max_abs_logprob" in x), default=None
        ),
        "divergences": [
            {"id": x["id"], "step": x["first_divergence"], "ref_margin": x["ref_margin_at_divergence"]}
            for x in present
            if not x["exact"]
        ],
    }
    for key in ("rendered_equal", "backend_prompt_ids_equal"):
        vals = [x[key] for x in present if key in x]
        if vals:
            summary[key] = f"{sum(vals)}/{len(vals)}"
    return {"reference": str(ref_dir), "candidate": str(cand_dir), "summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path, nargs="+")
    ap.add_argument("--out", type=Path, help="write the full comparison JSON here (one file per candidate dir name)")
    args = ap.parse_args(argv)
    for cand in args.candidate:
        res = compare(args.reference, cand)
        s = res["summary"]
        print(
            f"{cand.name}: exact {s['exact']}/{s['prompts'] - s['missing']}, token match "
            f"{s['token_match_fraction']:.3f}, finish equal {s['finish_equal']}, "
            f"step0 argmax {s['step0_argmax_equal']}, "
            f"step0 max|dlogp| {s['step0_max_abs_logprob']}, top5 {s['step0_top5_max_abs_logprob']}, "
            f"rendered {s.get('rendered_equal')}, backend ids {s.get('backend_prompt_ids_equal')}"
        )
        for d in s["divergences"]:
            print(f"    {d['id']:<20} diverges at step {d['step']} (ref margin {d['ref_margin']})")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"compare-{cand.name}.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
