"""Tests of `tools/qualification.py lock-candidate`, the snapshot locks of the qualification job `experimental`, and
of `lock-hosted`, the Kev bundle locks of tools/cuda-qualification-local.sh.

    python3 -I -m unittest discover -s tools/tests -v

Skipped where PyYAML (which tools/qualification.py imports) is not installed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "qualification.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
FILE = {"path": "config.json", "sha256": "ab" * 32, "size": 12}

try:
    import yaml  # noqa: F401
except ImportError:  # pragma: no cover - depends on the interpreter
    yaml = None


def _load():
    spec = importlib.util.spec_from_file_location("qualification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class LockCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qualification = _load()
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def run_main(self, origin: dict, *extra: str) -> tuple[int, dict | None, str]:
        profile = self.dir / "profile.json"
        profile.write_text(json.dumps({"id": "fam/name", "origin": origin}), encoding="utf-8")
        out = self.dir / "lock.json"
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            status = self.qualification.main(["lock-candidate", str(profile), str(out), *extra])
        lock = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
        return status, lock, err.getvalue()

    def test_snapshot_lock(self) -> None:
        status, lock, _ = self.run_main({"repo": "org/model", "revision": SHA, "files": [FILE]})
        self.assertEqual(status, 0)
        self.assertEqual(
            lock,
            {
                "schema_version": 1,
                "name": "fam-name-snapshot",
                "repository": "org/model",
                "revision": SHA,
                "files": [FILE],
            },
        )

    def test_base_lock(self) -> None:
        origin = {
            "repo": "org/model",
            "revision": SHA,
            "files": [FILE],
            "base": {"repo": "org/base", "revision": SHA, "files": [FILE]},
        }
        status, lock, _ = self.run_main(origin, "--base")
        self.assertEqual(status, 0)
        self.assertEqual((lock["repository"], lock["name"]), ("org/base", "fam-name-base-snapshot"))

    def test_refuses_branch_names_missing_base_and_extra_fields(self) -> None:
        self.assertEqual(self.run_main({"repo": "org/model", "revision": "main", "files": [FILE]})[0], 2)
        self.assertEqual(self.run_main({"repo": "org/model", "revision": SHA, "files": [FILE]}, "--base")[0], 2)
        extra = {**FILE, "url": "https://example.invalid/x"}
        self.assertEqual(self.run_main({"repo": "org/model", "revision": SHA, "files": [extra]})[0], 2)

    def test_every_experimental_candidate_locks(self) -> None:
        for name, base in (("kev-0.8b", True), ("qwen3", False), ("qwen25", False), ("whisper-large-v3-turbo", True)):
            profile = ROOT / "model-pipeline" / "candidates" / name / "profile.json"
            for extra in [[], ["--base"]] if base else [[]]:
                out = self.dir / f"{name}{'-base' if extra else ''}.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    status = self.qualification.main(["lock-candidate", str(profile), str(out), *extra])
                self.assertEqual(status, 0, f"{name} {extra}")


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class LockHostedTests(unittest.TestCase):
    """`lock-hosted`: the Kev bundle locks of tools/cuda-qualification-local.sh."""

    BUNDLE = ("model.onnx", "model.onnx.data", "head.safetensors", "head.json", "tokenizer.json")

    def setUp(self) -> None:
        self.qualification = _load()
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def run_main(self, manifest: Path, *paths: str) -> tuple[int, dict | None]:
        out = self.dir / "lock.json"
        out.unlink(missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = self.qualification.main(["lock-hosted", str(manifest), str(out), *paths])
        return status, json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None

    def test_kev_bundles_download_from_their_hosting_repositories(self) -> None:
        for name, repository, hosting in (
            ("kev-0.8b", "jaredpalmer/kev-0.8b", "Starlinghq/kev-0.8b-onnx"),
            ("kev-4b", "jaredpalmer/kev-4b", "Starlinghq/kev-4b-onnx"),
        ):
            for profile in ("onnx-fp32-cpu", "onnx-fp32-cuda"):
                manifest = ROOT / "model-pipeline" / "registry" / name / profile / "manifest.yaml"
                status, lock = self.run_main(manifest, *self.BUNDLE)
                self.assertEqual(status, 0, f"{name}/{profile}")
                self.assertEqual(lock["schema_version"], 2)
                self.assertEqual(lock["repository"], repository)
                self.assertEqual(tuple(f["path"] for f in lock["files"]), self.BUNDLE)
                for f in lock["files"]:
                    self.assertTrue(f["url"].startswith(f"https://huggingface.co/{hosting}/resolve/"), f["url"])
                    self.assertTrue(f["url"].endswith("/" + f["path"]), f["url"])

    def test_refuses_unknown_paths_and_manifests_without_a_source_base(self) -> None:
        kev = ROOT / "model-pipeline" / "registry" / "kev-0.8b" / "onnx-fp32-cuda" / "manifest.yaml"
        self.assertEqual(self.run_main(kev, "model.onnx", "not-in-origin-files.bin"), (2, None))
        manifest = self.dir / "manifest.yaml"
        origin = {"weights": {"url": "https://huggingface.co/org/model", "revision": SHA}, "files": [FILE]}
        manifest.write_text(json.dumps({"id": "fam/name/profile", "origin": origin}), encoding="utf-8")
        self.assertEqual(self.run_main(manifest, "config.json"), (2, None))


if __name__ == "__main__":
    unittest.main()
