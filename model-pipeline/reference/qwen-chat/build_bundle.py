"""Build flat-config Qwen2.5/Qwen3 CPU fp32 using the pinned GenAI builder.

Original assets are verified before use, copied to scratch, and never edited.
New graph digests remain observed candidate outputs requiring separate review.
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
from pathlib import Path, PurePosixPath

from common import write_json

from model_pipeline.onboarding_generation import CONFIG_CONTRACT

BUILDER_VERSION = "0.17.1"
BUILDER_SOURCE = "83de55a1f3886cddbb32a5517e26adb3f9d65159"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def float32_config(config: dict) -> dict:
    """Only dtype declarations change; these profiles have a flat text config."""
    if config.get("model_type") not in {"qwen2", "qwen3"} or "text_config" in config:
        raise ValueError("expected original flat Qwen2/Qwen3 text config")
    result = json.loads(json.dumps(config))
    result["dtype"] = "float32"
    result["torch_dtype"] = "float32"
    return result


def verified_copy(candidate: dict, snapshot: Path, target: Path) -> str:
    target.mkdir()
    paths = set()
    for record in candidate["origin"]["files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"] or record["path"] in paths:
            raise ValueError("unsafe or duplicate candidate asset path")
        paths.add(record["path"])
        source = snapshot / relative
        if not source.is_file() or source.stat().st_size != record["size"] or digest(source) != record["sha256"]:
            raise ValueError(f"asset mismatch: {record['path']}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.name == "config.json":
            config = json.loads(source.read_text(encoding="utf-8"))
            geometry = CONFIG_CONTRACT[config["model_type"]]
            for key, expected in geometry.items():
                actual = config.get(
                    key, config["hidden_size"] // config["num_attention_heads"] if key == "head_dim" else None
                )
                if actual != expected:
                    raise ValueError(f"original config {key} differs from admitted geometry")
            write_json(destination, float32_config(config))
        elif source.suffix == ".safetensors":
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        else:
            shutil.copyfile(source, destination)
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(paths):
        raise ValueError("candidate must pin config and tokenizer inputs")
    return digest(target / "config.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "snapshot", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if metadata.version("onnxruntime-genai") != BUILDER_VERSION:
        raise ValueError("wrong GenAI builder version")
    if args.out.resolve().is_relative_to(args.snapshot.resolve()) or args.snapshot.resolve().is_relative_to(
        args.out.resolve()
    ):
        raise ValueError("output overlaps original snapshot")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen-candidate-builder-") as temporary:
        work = Path(temporary)
        source = work / "snapshot-float32"
        config_digest = verified_copy(candidate, args.snapshot, source)
        built = work / "built"
        argv = [
            sys.executable,
            "-m",
            "onnxruntime_genai.models.builder",
            "-i",
            str(source),
            "-o",
            str(built),
            "-p",
            "fp32",
            "-e",
            "cpu",
            "-c",
            str(work / "cache"),
        ]
        with (args.out / "builder.log").open("w", encoding="utf-8") as stream:
            subprocess.run(argv, check=True, stdout=stream, stderr=subprocess.STDOUT)  # noqa: S603 - fixed argv
        for name in ("model.onnx", "model.onnx.data", "genai_config.json"):
            shutil.copyfile(built / name, args.out / name)
        # The owned tokenizer loads the original asset, not an implicit builder tokenizer.
        shutil.copyfile(args.snapshot / "tokenizer.json", args.out / "tokenizer.json")
        shutil.copyfile(args.snapshot / "tokenizer_config.json", args.out / "tokenizer_config.json")
        config = json.loads((args.snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = config.get("chat_template")
        if not isinstance(template, str):
            raise ValueError("expected a single pinned string chat_template")
        (args.out / "chat_template.jinja").write_text(template, encoding="utf-8")
        files = [
            {"path": name, "size": (args.out / name).stat().st_size, "sha256": digest(args.out / name)}
            for name in (
                "model.onnx",
                "model.onnx.data",
                "genai_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        ]
        write_json(
            args.out / "bundle-lock.json",
            {
                "format_version": 1,
                "profile": candidate["id"],
                "origin": candidate["origin"],
                "builder": {"version": BUILDER_VERSION, "source_commit": BUILDER_SOURCE},
                "float32_config_sha256": config_digest,
                "precision": "fp32",
                "provider": "CPUExecutionProvider",
                "files": files,
                "status": "proposed; observed graph bytes require independent review",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
