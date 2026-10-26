import copy
import json
from pathlib import Path

import pytest

from model_pipeline.asset_lock import lowest_version, validate, validate_file
from model_pipeline.cli import main

# Written by ModelAssetLock.ToJson; ai-dotnet/tests/Starlings.Acquisition.Tests checks that the .NET reader and writer
# round-trip these files byte for byte.
LOCKS = Path(__file__).parent / "data" / "asset-locks"
SHA = "0123456789abcdef0123456789abcdef01234567"


def load(name: str) -> dict:
    return json.loads((LOCKS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["v1-hub.json", "v2-sources.json", "v3-hosting.json"])
def test_committed_locks_are_valid_and_use_their_lowest_version(name):
    assert validate_file(LOCKS / name) == []
    lock = load(name)
    assert lock["schema_version"] == lowest_version(lock)


def test_cli_validates_locks(capsys):
    assert main(["validate-lock", *(str(p) for p in sorted(LOCKS.glob("*.json")))]) == 0
    assert "3 lock(s) checked, 0 invalid" in capsys.readouterr().out


def test_cli_reports_an_invalid_lock(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    assert main(["validate-lock", str(path)]) == 1
    assert "not a readable JSON document" in capsys.readouterr().err


def edited(name: str, edit) -> dict:
    lock = copy.deepcopy(load(name))
    edit(lock)
    return lock


@pytest.mark.parametrize(
    ("name", "edit"),
    [
        # Fields of a later version in an earlier one.
        ("v1-hub.json", lambda lock: lock["files"][0].update(url="https://example.test/f")),
        ("v2-sources.json", lambda lock: lock.update(display_name="Tiny")),
        ("v2-sources.json", lambda lock: lock.update(hosting={"repository": "c/d", "revision": SHA})),
        ("v1-hub.json", lambda lock: lock.update(schema_version=4)),
        # Pins and paths the .NET reader rejects.
        ("v1-hub.json", lambda lock: lock.update(revision="main")),
        ("v3-hosting.json", lambda lock: lock["hosting"].update(revision="v1.0")),
        ("v3-hosting.json", lambda lock: lock["hosting"].pop("revision")),
        ("v3-hosting.json", lambda lock: lock.update(display_name="  ")),
        ("v3-hosting.json", lambda lock: lock.update(display_name="a\nb")),
        ("v1-hub.json", lambda lock: lock.update(repository="acme")),
        ("v1-hub.json", lambda lock: lock.update(repository="../tiny")),
        ("v1-hub.json", lambda lock: lock["files"][0].update(path="../escape")),
        ("v1-hub.json", lambda lock: lock["files"][0].update(path="/abs")),
        ("v1-hub.json", lambda lock: lock["files"][0].update(path="a//b")),
        ("v1-hub.json", lambda lock: lock["files"][0].update(path="a\\b")),
        ("v1-hub.json", lambda lock: lock["files"][0].update(path="dir/./f")),
        ("v1-hub.json", lambda lock: lock["files"][0].update(sha256="A" * 64)),
        ("v1-hub.json", lambda lock: lock["files"][0].update(size=-1)),
        ("v1-hub.json", lambda lock: lock.update(files=[])),
        ("v1-hub.json", lambda lock: lock.update(extra=True)),
        ("v2-sources.json", lambda lock: lock["files"][0].update(url="http://example.test/f")),
        ("v2-sources.json", lambda lock: lock["files"][0].update(url="https://user:pw@example.test/f")),
        ("v2-sources.json", lambda lock: lock["files"][0].update(url="https://example.test/f#frag")),
        # Duplicate paths, compared without case as the .NET reader does.
        ("v1-hub.json", lambda lock: lock["files"][1].update(path="ONNX/model.onnx")),
    ],
)
def test_invalid_locks_are_rejected(name, edit):
    assert validate(edited(name, edit), Path(name))


def test_version_three_without_new_fields_is_still_valid():
    # A writer uses the lowest version, but a reader accepts a higher declared version without the newer fields.
    assert validate(edited("v1-hub.json", lambda lock: lock.update(schema_version=3)), Path("x")) == []
