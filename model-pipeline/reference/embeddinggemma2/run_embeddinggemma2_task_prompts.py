"""Reference embeddings of google/embeddinggemma-2 for every named prompt (capability `prompts_task`). CPU, fp32.

The accepted text fixtures cover the prompt settings none, `query` and `document` only. This runner embeds a few
public texts with each of the 20 names in `config_sentence_transformers.json` at the pinned revision, exactly as
sentence-transformers does when a caller passes `prompt_name=<name>` (the prompt text is prepended, the chat template
renders it, and the prompt tokens are pooled with the text's), with the same loading as `run_embeddinggemma2.py`
(`dtype=float32`, text path only, eager attention off by default, TF32 off).

Texts: passages from two public-domain Project Gutenberg files pinned by SHA-256 (the files and cleaning of
`model-pipeline/proposed/embeddinggemma-2/long-input/texts.json`: Alice's Adventures in Wonderland, eBook #11; the
United States Constitution, eBook #5) and one non-English case of the accepted text fixture.

Writes to `--out`:
- task-prompt-embeddings.jsonl: one row per (text, prompt name) in the layout of the protected `embeddings.jsonl`
  (id, prompt, prompt_text, text, input_ids, embedding, norm). `input_ids` are sentence-transformers' own
  preprocessing (`model.tokenize` on the prompted text), checked equal to `tokenizer.json` on `prompt + text`.
- environment.json: interpreter, packages, CPU, asset digests, the prompt table, and per-row checks (prompt text
  equals the configuration, batch vs single, names sharing a prompt text give equal embeddings).

Usage:
    python run_embeddinggemma2_task_prompts.py --model-dir <snapshot> --revision <sha> --sources <dir with pg11.txt,
        pg5.txt> --fixture <protected embeddings.jsonl> --out <dir> [--threads N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

REPO = "google/embeddinggemma-2"
SOURCES = {
    "pg11.txt": "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
    "pg5.txt": "d177168e3f4898213dced30fbbe60d74ad08e884ed5dd86bff94a1b8836b5982",
}
PACKAGES = ["torch", "transformers", "sentence-transformers", "tokenizers", "numpy"]
ASSETS = [
    "tokenizer.json",
    "config_sentence_transformers.json",
    "config.json",
    "model.safetensors",
    "chat_template.jinja",
]
# The fixture case reused (public: written for this repository's M0 corpus) for a non-English text.
FIXTURE_CASE = "chinese"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gutenberg_passage(path: Path, start: str, end: str | None) -> str:
    text = path.read_bytes().decode("utf-8").lstrip("﻿").replace("\r\n", "\n")
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j].strip()


def texts(sources: Path, fixture: Path) -> list[tuple[str, str]]:
    for name, want in SOURCES.items():
        have = sha256_file(sources / name)
        if have != want:
            raise SystemExit(f"{name}: SHA-256 {have}, expected {want}")
    alice = gutenberg_passage(sources / "pg11.txt", "\nCHAPTER I.\n", "*** END OF THE PROJECT GUTENBERG EBOOK")
    constitution = gutenberg_passage(
        sources / "pg5.txt", "THE CONSTITUTION OF THE UNITED STATES OF AMERICA, 1787", "William Jackson, Secretary"
    )
    paragraphs = [p.strip() for p in alice.split("\n\n") if p.strip()]
    rabbit = next(p for p in paragraphs if p.startswith("There was nothing so _very_ remarkable"))
    opening = next(p for p in paragraphs if p.startswith("Alice was beginning to get very tired"))
    preamble_start = constitution.index("We the people")
    preamble = " ".join(constitution[preamble_start : constitution.index("\n\n", preamble_start)].split())
    row = next(json.loads(line) for line in fixture.open(encoding="utf-8") if json.loads(line)["id"] == FIXTURE_CASE)
    return [
        ("alice-opening-clause", " ".join(opening.split(":")[0].split())),
        ("alice-rabbit-paragraph", " ".join(rabbit.split())),
        ("constitution-preamble", preamble),
        (f"fixture-{FIXTURE_CASE}", row["text"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.model_dir / "config_sentence_transformers.json").read_text(encoding="utf-8"))
    prompts: dict[str, str] = config["prompts"]
    if config.get("default_prompt_name") is not None or len(prompts) != 20:
        raise SystemExit("unexpected prompt configuration")
    model = SentenceTransformer(str(args.model_dir), device="cpu", model_kwargs={"dtype": torch.float32})
    model.eval()
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    rows, checks = [], []
    cases = texts(args.sources, args.fixture)
    with torch.inference_mode():
        for case_id, text in cases:
            by_text: dict[str, np.ndarray] = {}
            for name, prompt_text in prompts.items():
                emb = model.encode([text], prompt_name=name, convert_to_numpy=True, batch_size=1)[0].astype(np.float32)
                features = model.tokenize([prompt_text + text])
                ids = features["input_ids"][0][features["attention_mask"][0].bool()].tolist()
                ref_ids = tok.encode(prompt_text + text).ids
                if ids != ref_ids:
                    raise SystemExit(f"{case_id}@{name}: sentence-transformers ids differ from tokenizer.json")
                if prompt_text in by_text:
                    same = float(np.max(np.abs(by_text[prompt_text] - emb)))
                    checks.append({"row": f"{case_id}@{name}", "same_prompt_text_max_abs": same})
                by_text.setdefault(prompt_text, emb)
                rows.append(
                    {
                        "id": case_id,
                        "prompt": name,
                        "prompt_text": prompt_text,
                        "text": text,
                        "input_ids": ids,
                        "embedding": [float(x) for x in emb],
                        "norm": float(np.linalg.norm(emb.astype(np.float64))),
                    }
                )
            # Padded batch of all 20 prompted inputs vs the single rows.
            prompted = model.encode(
                [p + text for p in prompts.values()], convert_to_numpy=True, batch_size=len(prompts)
            ).astype(np.float32)
            singles = np.array([r["embedding"] for r in rows[-len(prompts) :]], np.float32)
            checks.append(
                {"case": case_id, "padded_batch_vs_single_max_abs": float(np.max(np.abs(prompted - singles)))}
            )
            print(case_id, len(ids), flush=True)

    with (args.out / "task-prompt-embeddings.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    env = {
        "runner": "model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_task_prompts.py",
        "repository_commit": commit,
        "model": {"repo": REPO, "revision": args.revision},
        "assets": {a: sha256_file(args.model_dir / a) for a in ASSETS},
        "sources": SOURCES,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
        "cpu": platform.processor() or platform.machine(),
        "threads": args.threads,
        "dtype": "float32",
        "device": "cpu",
        "prompts": prompts,
        "rows": len(rows),
        "checks": checks,
        "max_same_prompt_text_max_abs": max(c.get("same_prompt_text_max_abs", 0.0) for c in checks),
        "max_padded_batch_vs_single_max_abs": max(c.get("padded_batch_vs_single_max_abs", 0.0) for c in checks),
    }
    (args.out / "environment.json").write_text(json.dumps(env, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
