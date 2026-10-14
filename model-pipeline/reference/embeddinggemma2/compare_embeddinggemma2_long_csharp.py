"""Compare the C# package on the proposed long-input rows with the reference (W3, EmbeddingGemma 2, long inputs).

Reads the reference rows (`long-embeddings.jsonl` of `run_embeddinggemma2_long.py`) and the C# outputs
(`long.results.json` of `Starlings.Fidelity.Tool embeddinggemma2-long`) and reports, per row and worst: rendered ids
equal, the outcome against the row's expectation (`embed`, or `reject` above 8,192 tokens with
EmbeddingGemma2InputTooLongException), embedding max abs and cosine (float64), NaN/Inf, and padded batches against the
C# single rows. The bounds in the report are the existing protected values of the matching profiles, shown for
orientation only: these rows are proposed, not accepted fixtures, and no bound applies to them yet.

Usage:
    python compare_embeddinggemma2_long_csharp.py --reference <long-embeddings.jsonl> --run <long.results.json>
        --out <report.json> [--markdown <report.md>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# The protected bounds of embeddinggemma-2/onnx-fp32-cpu (text rows) and video-fp32-cpu (media rows), for orientation.
ORIENTATION = {"text": {"embedding_max_abs": 1e-5}, "media": {"embedding_max_abs": 1e-5, "min_cosine": 0.9999995}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    lines = args.reference.read_text(encoding="utf-8").split("\n")
    ref = {r["id"]: r for r in (json.loads(line) for line in lines if line.strip())}
    run = json.loads(args.run.read_text())
    if run.get("rows_file_sha256") != hashlib.sha256(args.reference.read_bytes()).hexdigest():
        raise SystemExit("the C# run was made on another long-embeddings.jsonl")
    rows, singles = {}, {}
    worst = {
        k: {"embedding_max_abs": 0.0, "worst_row": None, "min_cosine": 1.0, "rows": 0} for k in ("text", "media", "all")
    }
    for rid, r in ref.items():
        c = run["rows"].get(rid)
        if c is None:
            rows[rid] = {"missing": True}
            continue
        kind = "text" if r["kind"] == "text" else "media"
        rec = {
            "kind": kind,
            "tokens": r["tokens"],
            "expect": r["expect"],
            "outcome": c["outcome"],
            "matches_expectation": (r["expect"] == "embed") == (c["outcome"] == "embedded"),
            # Rows the package rejects at tokenization have no ids; their token count is in the exception instead.
            "ids_equal": c.get("input_ids") == r["input_ids"] if "input_ids" in c else None,
            "seconds": c.get("seconds"),
        }
        if c["outcome"] == "embedded":
            e = np.array(c["embedding"], np.float64)
            x = np.array(r["embedding"], np.float64)
            singles[rid] = e
            d = float(np.abs(e - x).max())
            cos = float(e @ x / np.linalg.norm(e) / np.linalg.norm(x))
            rec.update(embedding_max_abs=d, cosine=cos, finite=bool(np.isfinite(e).all()))
            if r["expect"] == "embed":
                for k in (kind, "all"):
                    w = worst[k]
                    w["rows"] += 1
                    if d > w["embedding_max_abs"]:
                        w["embedding_max_abs"], w["worst_row"] = d, rid
                    w["min_cosine"] = min(w["min_cosine"], cos)
        else:
            rec["exception"] = c.get("exception")
            count = (c.get("exception") or {}).get("token_count")
            rec["rejected_token_count_equal"] = count == r["tokens"]
        rows[rid] = rec
    batches = {}
    for name, b in run.get("batches", {}).items():
        if "error" in b:  # the run recorded why the batch did not run (for example GPU memory)
            batches[name] = {"rows": b["rows"], "not_run": b["error"]}
            continue
        pairs = zip(b["rows"], b["embeddings"], strict=True)
        diffs = [float(np.abs(np.array(e, np.float64) - singles[rid]).max()) for rid, e in pairs]
        batches[name] = {"rows": b["rows"], "max_abs_vs_single": max(diffs)}
    report = {
        "reference": str(args.reference.name),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate": run.get("candidate"),
        "candidate_commit": run.get("repository_commit"),
        "environment": run.get("environment"),
        "rows": len(ref),
        "all_ids_equal": all(v.get("ids_equal") is not False for v in rows.values()),
        "all_rejected_token_counts_equal": all(v.get("rejected_token_count_equal", True) for v in rows.values()),
        "all_outcomes_as_expected": all(v.get("matches_expectation") for v in rows.values()),
        "all_finite": all(v.get("finite", True) for v in rows.values()),
        "worst": worst,
        "batches": batches,
        "orientation_only_existing_bounds": ORIENTATION,
        "per_row": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(report))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_row", "environment")}, indent=1))
    return 0 if report["all_ids_equal"] and report["all_outcomes_as_expected"] and report["all_finite"] else 1


def markdown(report: dict) -> str:
    lines = [
        "| Row | Tokens | Expect | C# outcome | Ids equal | Embedding max abs | Cosine | Seconds |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for rid, r in report["per_row"].items():
        if r.get("missing"):
            lines.append(f"| `{rid}` | | | missing | | | | |")
            continue
        d = f"{r['embedding_max_abs']:.2e}" if "embedding_max_abs" in r else ""
        c = f"{r['cosine']:.12f}" if "cosine" in r else ""
        outcome = r["outcome"] + (
            f" ({r['exception']['type']}, {r['exception']['token_count']} tokens)" if r.get("exception") else ""
        )
        ids = {True: "yes", False: "no", None: "n/a (rejected)"}[r["ids_equal"]]
        lines.append(f"| `{rid}` | {r['tokens']} | {r['expect']} | {outcome} | {ids} | {d} | {c} | {r['seconds']} |")
    lines.append("")
    for k, w in report["worst"].items():
        lines.append(
            f"- Worst ({k} rows, {w['rows']} embedded): embedding max abs {w['embedding_max_abs']:.2e} "
            f"(`{w['worst_row']}`), min cosine {w['min_cosine']:.12f}"
        )
    for name, b in report["batches"].items():
        if "not_run" in b:
            lines.append(f"- Padded batch `{name}` ({', '.join(b['rows'])}): not run ({b['not_run']})")
            continue
        lines.append(
            f"- Padded batch `{name}` ({', '.join(b['rows'])}) vs C# single rows: max abs {b['max_abs_vs_single']:.2e}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
