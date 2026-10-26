"""Tests of `tools/qualification.py lock-candidate`, the snapshot locks of the qualification job `experimental`.

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


if __name__ == "__main__":
    unittest.main()
