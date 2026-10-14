"""Upstream change detection (P-020). Recorded responses only: no test opens a network connection."""

import json
import urllib.request
from pathlib import Path

import pytest
import yaml

from model_pipeline import upstream
from model_pipeline.cli import main
from model_pipeline.manifest import find_manifests, load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"
REGISTRY = Path(__file__).resolve().parents[1] / "registry"

PINNED = "1" * 40
NEWER = "2" * 40
KNOWN = "3" * 40
REPO = "example-org/encoder"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def write_manifest(tmp_path: Path, known: str | None = None, weights_url: str | None = None) -> Path:
    manifest = load(SYNTHETIC)
    manifest.pop("synthetic")
    manifest["id"] = "example/fp32-cpu"
    url = weights_url or f"https://huggingface.co/{REPO}"
    manifest["origin"]["code"] = {"url": url, "revision": PINNED}
    manifest["origin"]["weights"] = {"url": url, "revision": PINNED}
    manifest["origin"]["files"] = [
        {"path": p, "sha256": "0" * 64, "size": 1} for p in ("README.md", "tokenizer.json", "onnx/model.onnx")
    ]
    manifest["maintenance"]["latest_verified_upstream_revision"] = PINNED
    if known:
        manifest["maintenance"]["latest_known_upstream_revision"] = known
    path = tmp_path / "registry" / "example" / "fp32-cpu" / "manifest.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def info(sha: str, **files: str) -> dict:
    return {"sha": sha, "siblings": [{"rfilename": k.replace("__", "/"), "blobId": v} for k, v in files.items()]}


BASE = {"README.md": "a", "tokenizer.json": "b", "onnx__model.onnx": "c", "extra.txt": "d"}


def run(tmp_path: Path, responses: dict, *extra: str) -> tuple[int, dict]:
    responses_path = tmp_path / "responses.json"
    responses_path.write_text(json.dumps(responses), encoding="utf-8")
    out = tmp_path / "report.json"
    code = main(
        ["upstream", str(tmp_path / "registry"), "--responses", str(responses_path), "--json", str(out), *extra]
    )
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_current_revision_reports_nothing_new(tmp_path):
    write_manifest(tmp_path, known=PINNED)
    code, report = run(tmp_path, {REPO: {"main": info(PINNED, **BASE)}})
    assert code == 0
    [source] = report["sources"]
    assert source["status"] == "current" and source["role"] == "code+weights" and source["tracked"]
    assert report["summary"]["current"] == 1


def test_changed_tokenizer_and_card_scope_the_gates(tmp_path, capsys):
    write_manifest(tmp_path, known=PINNED)
    changed = dict(BASE, **{"README.md": "a2", "tokenizer.json": "b2", "extra.txt": "d2"})
    code, report = run(tmp_path, {REPO: {"main": info(NEWER, **changed), PINNED: info(PINNED, **BASE)}})
    assert code == 1
    [source] = report["sources"]
    assert source["status"] == "changed" and source["new_since_known"]
    assert source["gates"] == ["G0", "G1", "G2", "G3", "G6"]
    kinds = {c["path"]: (c["kind"], c["locked"]) for c in source["changed_files"]}
    assert kinds == {"README.md": ("card", True), "tokenizer.json": ("tokenizer", True), "extra.txt": ("config", False)}
    out = capsys.readouterr().out
    assert f"set `maintenance.latest_known_upstream_revision` to `{NEWER}`" in out


def test_changed_weights_add_resource_and_performance_gates(tmp_path):
    write_manifest(tmp_path)
    changed = dict(BASE, **{"onnx__model.onnx": "c2"})
    _, report = run(tmp_path, {REPO: {"main": info(NEWER, **changed), PINNED: info(PINNED, **BASE)}})
    assert report["sources"][0]["gates"] == ["G0", "G2", "G3", "G4", "G5", "G6"]


def test_revision_already_known_is_not_new(tmp_path):
    write_manifest(tmp_path, known=NEWER)
    code, report = run(tmp_path, {REPO: {"main": info(NEWER, **BASE), PINNED: info(PINNED, **BASE)}})
    assert code == 0
    [source] = report["sources"]
    assert source["status"] == "changed" and not source["new_since_known"]


def test_only_unlocked_files_changed_needs_no_gate(tmp_path):
    write_manifest(tmp_path)
    changed = dict(BASE, **{"extra.txt": "d2"})
    _, report = run(tmp_path, {REPO: {"main": info(NEWER, **changed), PINNED: info(PINNED, **BASE)}})
    [source] = report["sources"]
    assert source["gates"] == [] and "pin can stay" in source["detail"]


def test_removed_locked_file_is_reported(tmp_path):
    write_manifest(tmp_path)
    changed = {k: v for k, v in BASE.items() if k != "tokenizer.json"}
    _, report = run(tmp_path, {REPO: {"main": info(NEWER, **changed), PINNED: info(PINNED, **BASE)}})
    [change] = report["sources"][0]["changed_files"]
    assert change == {"path": "tokenizer.json", "change": "removed", "kind": "tokenizer", "locked": True}


def test_unknown_pinned_listing_scopes_every_gate(tmp_path):
    write_manifest(tmp_path)
    _, report = run(tmp_path, {REPO: {"main": info(NEWER, **BASE), PINNED: {"error": "HTTP 404"}}})
    assert report["sources"][0]["gates"] == list(upstream.ALL_GATES)


def test_fetch_error_exits_2(tmp_path):
    write_manifest(tmp_path)
    code, report = run(tmp_path, {REPO: {"main": {"error": "HTTP 401"}}})
    assert code == 2
    assert report["sources"][0]["status"] == "error" and "HTTP 401" in report["sources"][0]["detail"]


def test_non_hub_source_is_unchecked(tmp_path):
    write_manifest(tmp_path, weights_url="https://github.com/example-org/encoder")
    code, report = run(tmp_path, {})
    assert code == 0 and report["sources"][0]["status"] == "unchecked"


@pytest.mark.parametrize(
    ("url", "repo"),
    [
        ("https://huggingface.co/intfloat/multilingual-e5-small", "intfloat/multilingual-e5-small"),
        ("https://huggingface.co/datasets/x/y", None),
        ("http://huggingface.co/x/y", None),
        ("https://github.com/fastino-ai/GLiNER2", None),
        ("https://huggingface.co.evil.example/x/y", None),
    ],
)
def test_hf_repo(url, repo):
    assert upstream.hf_repo(url) == repo


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("tokenizer.json", "tokenizer"),
        ("sentencepiece.bpe.model", "tokenizer"),
        ("chat_template.jinja", "tokenizer"),
        ("onnx/model.onnx", "weights"),
        ("model.safetensors", "weights"),
        ("README.md", "card"),
        ("config.json", "config"),
        ("1_Pooling/config.json", "config"),
    ],
)
def test_classify(path, kind):
    assert upstream.classify(path) == kind


def test_registry_with_every_pin_current(tmp_path):
    """The real registry parses, and recorded responses equal to the pins report nothing new."""
    responses: dict = {}
    for path in find_manifests(REGISTRY):
        origin = load(path)["origin"]
        for role in ("code", "weights"):
            repo = upstream.hf_repo(origin[role]["url"])
            if repo:
                responses[repo] = {"main": {"sha": origin[role]["revision"], "siblings": []}}
    report = upstream.check_registry(REGISTRY, upstream.responses_fetcher(responses), today="2026-10-13")
    assert report["summary"]["error"] == 0 and report["summary"]["changed"] == 0
    assert report["summary"]["current"] > 0 and report["summary"]["new_since_known"] == 0
