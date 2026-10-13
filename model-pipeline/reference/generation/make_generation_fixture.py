"""Write the pilot LLM generation fixture from a ``run_reference.py`` output directory (backlog P-016).

The fixture (``generation.jsonl``, fixture format ``starlings.generation-fixture`` version 1) holds, per prompt
of ``prompts-v1.jsonl``, the prompt itself and the reference's rendering, prompt ids, greedy output ids (including the
stop token), finish reason, decoded text, per-step chosen log-probability and top-1 minus top-2 logit margin, and the
step-0 log-softmax restricted to its 64 most likely ids. The full step-0 vectors (22 x 248,320 float32, 22 MB) are not
copied; their SHA-256 digests are recorded so a full-vector comparison can prove it used the same reference.

Usage::

    python make_generation_fixture.py --reference <run_reference.py --out dir> --prompts prompts-v1.jsonl --out <dir>

Writes ``<out>/generation.jsonl`` and ``<out>/environment.json`` (the runner's environment plus the fixture's
provenance). Needs numpy (the reference environment has it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

FORMAT = "starlings.generation-fixture"
FORMAT_VERSION = 1
TOP_K = 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def git_state(repo: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()  # noqa: S603, S607

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    prompts = read_jsonl(args.prompts)
    results = {r["id"]: r for r in read_jsonl(args.reference / "results.jsonl")}
    env = json.loads((args.reference / "environment.json").read_text(encoding="utf-8"))
    if env.get("prompts_sha256") != sha256_file(args.prompts):
        raise SystemExit("the reference run used a different prompts file")

    args.out.mkdir(parents=True, exist_ok=True)
    lines = []
    step0_digests = {}
    for p in prompts:
        r = results[p["id"]]
        vec_path = args.reference / "step0" / f"{p['id']}.f32"
        vec = np.fromfile(vec_path, dtype="<f4")
        top = np.argsort(-vec, kind="stable")[:TOP_K]
        step0_digests[p["id"]] = sha256_file(vec_path)
        row = {
            "id": p["id"],
            "category": p["category"],
            "messages": p["messages"],
            "tools": p.get("tools"),
            "chat_template_kwargs": p.get("chat_template_kwargs"),
            "max_new_tokens": p["max_new_tokens"],
            "rendered": r["rendered"],
            "prompt_ids": r["prompt_ids"],
            "output_ids": r["output_ids"],
            "finish_reason": r["finish_reason"],
            "output_text": r["output_text"],
            "step_logprob": r["step_logprob"],
            "step_margin": r["step_margin"],
            "step0_top": [[int(i), float(vec[i])] for i in top],
            "step0_vocab": int(vec.shape[0]),
            "step0_sha256": step0_digests[p["id"]],
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    (args.out / "generation.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    repo = Path(__file__).resolve().parents[3]
    env_out = {
        "fixture_format": FORMAT,
        "fixture_format_version": FORMAT_VERSION,
        "generated_by": "model-pipeline/reference/generation/make_generation_fixture.py",
        "repository": git_state(repo),
        "prompts": {
            "path": "model-pipeline/reference/generation/prompts-v1.jsonl",
            "sha256": sha256_file(args.prompts),
        },
        "step0_top_k": TOP_K,
        "step0_full_vectors": {
            "note": "float32 little-endian log-softmax over the full vocabulary, kept outside Git",
            "sha256": step0_digests,
        },
        "reference_run": env,
    }
    (args.out / "environment.json").write_text(
        json.dumps(env_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(lines)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
