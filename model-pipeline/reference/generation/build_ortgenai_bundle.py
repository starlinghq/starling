"""Build the ONNX Runtime GenAI bundle of the pilot LLM reproducibly (backlog P-016).

Recipe (``docs/m0/generation-backend-check.md`` sections 2 and 9, kickoff "Generation backend"):

1. Verify the ``Qwen/Qwen3.5-0.8B`` snapshot at ``2fc06364715b967f1860aea9cf38778875588b17`` (files the build reads).
2. Make a copy whose ``config.json`` declares float32 (weights hard-linked, nothing else changed). Without it the
   0.17.1 builder computes the RMSNorm ``1 + w`` offsets and ``-exp(A_log)`` in bfloat16 before writing them as
   fp32, and fp32 matches the reference on only 16 of 22 prompts.
3. Run the onnxruntime-genai 0.17.1 builder (``rel-0.17.1`` head ``83de55a1f3886cddbb32a5517e26adb3f9d65159``) with
   ``--extra_options exclude_mtp=true``: variant ``fp32x`` is ``-p fp32 -e cpu`` (used on the CPU EP and, with
   ``use_tf32=0``, on the CUDA EP; ``-p fp32 -e cuda`` builds a graph that fails, M0 failure 2) and ``fp16x`` is
   ``-p fp16 -e cuda``.
4. Lay the files out as the C# asset lock expects (``Qwen35ModelAssets``), relative to the snapshot directory:
   ``ortgenai-<variant>/{genai_config.json,model.onnx,model.onnx.data}``, ``transformers-5.18.0/tokenizer.json``
   (the builder's copy of the transformers 5.18.0 runtime tokenizer, equal to the ``qwen35-bpe-chat`` profile asset)
   and ``chat_template.jinja`` (the snapshot's file; the C# renderer was ported from it).
5. Hash every output and compare with the pinned digests below; exit 1 on any difference. The builder output is
   used unmodified: runtime settings (pad id, provider, TF32) are overlays applied by the C# runtime, never edits.

Run in the environment of ``requirements.txt`` (torch 2.14.1, transformers 5.18.0, onnxruntime-genai 0.17.1,
onnxruntime 1.30.0, onnx 1.23.1). Standard library plus the builder module. Example::

    python build_ortgenai_bundle.py --snapshot <dir with Qwen3.5-0.8B@2fc06364> --out <bundle dir> --variant fp32x
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

REPOSITORY = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BUILDER_VERSION = "0.17.1"
BUILDER_SOURCE_COMMIT = "83de55a1f3886cddbb32a5517e26adb3f9d65159"

# Snapshot files the build reads (docs/m0/hub-pins.md, "Pilot LLM").
SNAPSHOT = {
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "model.safetensors.index.json": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
}
# config.json after the float32 overlay (step 2), serialized as below.
FLOAT32_CONFIG_SHA256 = "295154cd6039372b0b16a30336b11deea6591320bf21ee4dc053a305684da311"

VARIANTS = {
    "fp32x": {"precision": "fp32", "ep": "cpu"},
    "fp16x": {"precision": "fp16", "ep": "cuda"},
}

# Pinned outputs (bundle-relative path -> (sha256, size)), equal to the M0 builds (docs/m0/generation-backend-check.md).
EXPECTED = {
    "fp32x": {
        "ortgenai-fp32x/genai_config.json": ("e7e1957b2544fdad6170a9cc81cbbb1b88c8819b7016a609170ad7cbebba81ba", 1935),
        "ortgenai-fp32x/model.onnx": ("4a052c4a1444c9437e00ffd48baecb0838d3d9ca1703c3153dbd46e945e84703", 308301),
        "ortgenai-fp32x/model.onnx.data": (
            "129c58a320b6de7192713cc9c7e4ba744dcf7fabb0f57c6535021b45b3dc5143",
            3077177344,
        ),
    },
    "fp16x": {
        "ortgenai-fp16x/genai_config.json": ("ded164b67284883a13c6454d7445f814014a1e12277ce323011683cd6a69d055", 2109),
        "ortgenai-fp16x/model.onnx": ("26895023e4be1f528aef04ce3a07a14e70a2421cfb400a49fec639ea9a14f46d", 316521),
        "ortgenai-fp16x/model.onnx.data": (
            "47f6ee15d95080acc66f7c4acdf4012c83b20063cf0cfeb59a6b7ae09950a3b2",
            1572143104,
        ),
    },
}
SHARED = {
    "transformers-5.18.0/tokenizer.json": (
        "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
        19989339,
    ),
    "chat_template.jinja": ("273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80", 7755),
}
GRAPH_FILES = ("genai_config.json", "model.onnx", "model.onnx.data")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def float32_config(config: dict) -> dict:
    """The float32 overlay: only the dtype declarations change."""
    cfg = json.loads(json.dumps(config))
    cfg["dtype"] = "float32"
    cfg["torch_dtype"] = "float32"
    cfg["text_config"]["dtype"] = "float32"
    cfg.get("vision_config", {})["dtype"] = "float32"
    return cfg


def make_float32_copy(snapshot: Path, dst: Path) -> str:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(snapshot.iterdir()):
        if not f.is_file() or f.name in ("config.json", "hashes.json"):
            continue
        target = dst / f.name
        if f.suffix == ".safetensors":
            try:
                os.link(f, target)
            except OSError:
                shutil.copy2(f, target)
        else:
            shutil.copy2(f, target)
    cfg = float32_config(json.loads((snapshot / "config.json").read_text(encoding="utf-8")))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return sha256_file(dst / "config.json")


def run_builder(src: Path, out: Path, cache: Path, precision: str, ep: str, log: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-i",
        str(src),
        "-o",
        str(out),
        "-p",
        precision,
        "-e",
        ep,
        "-c",
        str(cache),
        "--extra_options",
        "exclude_mtp=true",
    ]
    print("+", " ".join(cmd), flush=True)
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)  # noqa: S603 - fixed argument list


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=Path, help=f"{REPOSITORY} snapshot at {REVISION}")
    ap.add_argument("--out", required=True, type=Path, help="bundle directory (the C# snapshot directory layout)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), action="append", required=True)
    ap.add_argument("--work", type=Path, help="scratch directory (default: a temporary directory)")
    args = ap.parse_args(argv)

    builder_version = metadata.version("onnxruntime-genai")
    if builder_version != BUILDER_VERSION:
        print(f"onnxruntime-genai {builder_version} installed; the recipe pins {BUILDER_VERSION}", file=sys.stderr)
        return 2

    for name, digest in SNAPSHOT.items():
        actual = sha256_file(args.snapshot / name)
        if actual != digest:
            print(f"snapshot {name}: sha256 {actual}, expected {digest}", file=sys.stderr)
            return 2
    print(f"ok: snapshot {REPOSITORY}@{REVISION} ({len(SNAPSHOT)} files verified)")

    work_ctx = tempfile.TemporaryDirectory(prefix="ortgenai-build-") if args.work is None else None
    work = Path(work_ctx.name) if work_ctx else args.work
    work.mkdir(parents=True, exist_ok=True)
    f32 = work / "snapshot-float32"
    cfg_digest = make_float32_copy(args.snapshot, f32)
    if cfg_digest != FLOAT32_CONFIG_SHA256:
        print(f"float32 config.json sha256 {cfg_digest}, expected {FLOAT32_CONFIG_SHA256}", file=sys.stderr)
        return 1
    print(f"ok: float32 config overlay {cfg_digest}")

    args.out.mkdir(parents=True, exist_ok=True)
    expected: dict[str, tuple[str, int]] = dict(SHARED)
    built_tokenizer = None
    for variant in args.variant:
        spec = VARIANTS[variant]
        build_dir = work / f"build-{variant}"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        run_builder(f32, build_dir, work / "cache", spec["precision"], spec["ep"], work / f"build-{variant}.log")
        dest = args.out / f"ortgenai-{variant}"
        dest.mkdir(parents=True, exist_ok=True)
        for name in GRAPH_FILES:
            shutil.copy2(build_dir / name, dest / name)
        built_tokenizer = build_dir / "tokenizer.json"
        if sha256_file(build_dir / "chat_template.jinja") != SHARED["chat_template.jinja"][0]:
            print(f"{variant}: the builder's chat_template.jinja differs from the snapshot's", file=sys.stderr)
            return 1
        expected.update(EXPECTED[variant])

    tok_dir = args.out / "transformers-5.18.0"
    tok_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_tokenizer, tok_dir / "tokenizer.json")
    shutil.copy2(args.snapshot / "chat_template.jinja", args.out / "chat_template.jinja")

    record = {
        "recipe": "model-pipeline/reference/generation/build_ortgenai_bundle.py",
        "repository": REPOSITORY,
        "revision": REVISION,
        "builder": {"package": "onnxruntime-genai", "version": builder_version, "source_commit": BUILDER_SOURCE_COMMIT},
        "packages": {
            p: metadata.version(p) for p in ("torch", "transformers", "onnxruntime", "onnx", "onnx-ir", "numpy")
        },
        "python": sys.version.split()[0],
        "float32_config_sha256": cfg_digest,
        "variants": {v: VARIANTS[v] for v in args.variant},
        "extra_options": "exclude_mtp=true",
        "files": {},
    }
    status = 0
    for rel, (digest, size) in sorted(expected.items()):
        path = args.out / rel
        actual, actual_size = sha256_file(path), path.stat().st_size
        ok = actual == digest and actual_size == size
        record["files"][rel] = {"sha256": actual, "size": actual_size, "matches_pin": ok}
        print(f"{'ok  ' if ok else 'DIFF'} {rel} {actual} {actual_size}")
        if not ok:
            status = 1
    (args.out / "build-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if work_ctx:
        work_ctx.cleanup()
    print("all outputs match the pinned digests" if status == 0 else "OUTPUTS DIFFER from the pinned digests")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
