"""Admission/source/environment negatives independent of real model execution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from model_pipeline import onboarding_candidate as candidate
from model_pipeline.onboarding_runner import PipelineError, Stage, run_pipeline


def contract(files):
    return {
        "format_version": 1,
        "id": "unit-candidate",
        "task": "embeddings",
        "origin": {"repo": "test/model", "revision": "a" * 40, "files": files},
        "runtime": {"backend": "onnx", "version": "1.30.0", "provider": "CPUExecutionProvider", "precision": "fp32"},
        "behavior": {"dimension": 4},
        "evaluation": {"status": "proposed", "policy": {"max_abs": 0.001}},
    }


def source(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    files = []
    for name, contents in (("config.json", b"{}\n"), ("model.weights", b"synthetic weights")):
        path = snapshot / name
        path.write_bytes(contents)
        files.append({"path": name, "size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()})
    return snapshot, contract(files)


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a\\b", "./asset", "a//asset"])
def test_loader_rejects_unsafe_or_noncanonical_asset_paths(tmp_path, path):
    _snapshot, data = source(tmp_path)
    data["origin"]["files"][0]["path"] = path
    profile = tmp_path / "profile.json"
    save(profile, data)
    with pytest.raises(PipelineError, match="unsafe"):
        candidate.load_candidate(profile)


def test_loader_rejects_duplicate_source_paths(tmp_path):
    _snapshot, data = source(tmp_path)
    data["origin"]["files"][1] = copy.deepcopy(data["origin"]["files"][0])
    profile = tmp_path / "profile.json"
    save(profile, data)
    with pytest.raises(PipelineError, match="duplicate"):
        candidate.load_candidate(profile)


@pytest.mark.parametrize("field,value", [("size", True), ("size", -1), ("sha256", "a" * 63), ("sha256", "A" * 64)])
def test_loader_rejects_malformed_source_records(tmp_path, field, value):
    _snapshot, data = source(tmp_path)
    data["origin"]["files"][0][field] = value
    profile = tmp_path / "profile.json"
    save(profile, data)
    with pytest.raises(PipelineError, match="invalid candidate"):
        candidate.load_candidate(profile)


@pytest.mark.parametrize(
    "alter",
    [
        lambda data: data.update({"unknown": 1}),
        lambda data: data["origin"].update({"revision": "main"}),
        lambda data: data["runtime"].update({"provider": "CUDAExecutionProvider"}),
        lambda data: data["evaluation"].update({"status": "adopted"}),
    ],
)
def test_loader_rejects_unknown_or_unpinned_contracts(tmp_path, alter):
    _snapshot, data = source(tmp_path)
    alter(data)
    profile = tmp_path / "profile.json"
    save(profile, data)
    with pytest.raises(PipelineError, match="invalid candidate"):
        candidate.load_candidate(profile)


def test_loader_bounds_contract_size(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(PipelineError, match="1 MiB"):
        candidate.load_candidate(path)


@pytest.mark.parametrize("damage", ["missing", "same-size-change", "different-size", "directory"])
def test_asset_verification_rejects_missing_or_corrupt_source(tmp_path, damage):
    snapshot, data = source(tmp_path)
    path = snapshot / "config.json"
    if damage == "missing":
        path.unlink()
    elif damage == "same-size-change":
        path.write_bytes(b"[]\n")
    elif damage == "different-size":
        path.write_bytes(b"different size")
    else:
        path.unlink()
        path.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(PipelineError):
        candidate.verify_assets(data, snapshot, out)
    assert not (out / "source-lock.json").exists()


def test_asset_verification_rejects_external_file_symlink(tmp_path):
    snapshot, data = source(tmp_path)
    external = tmp_path / "outside.json"
    external.write_bytes((snapshot / "config.json").read_bytes())
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(external)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(PipelineError, match="escapes snapshot"):
        candidate.verify_assets(data, snapshot, out)


def test_asset_verification_rejects_symlink_directory_even_inside_snapshot(tmp_path):
    snapshot, data = source(tmp_path)
    directory = snapshot / "actual"
    directory.mkdir()
    (directory / "config.json").write_bytes((snapshot / "config.json").read_bytes())
    (snapshot / "alias").symlink_to(directory, target_is_directory=True)
    data["origin"]["files"][0]["path"] = "alias/config.json"
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(PipelineError, match="symlink directory"):
        candidate.verify_assets(data, snapshot, out)


def test_successful_asset_lock_preserves_proposed_status_and_exact_identity(tmp_path):
    snapshot, data = source(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = candidate.verify_assets(data, snapshot, out)
    lock = json.loads((out / "source-lock.json").read_text())
    assert result == {"source_files_verified": 2, "reference_status": "proposed"}
    assert lock["profile"] == data["id"]
    assert lock["origin"] == data["origin"]
    assert all(record["verification"] == "pinned-size-and-sha256" for record in lock["assets"])


def setup_stages(tmp_path, monkeypatch, *, packages=None, python_status="available", dotnet_status="available"):
    root = tmp_path / "repo"
    root.mkdir()
    snapshot, data = source(tmp_path)
    profile = root / "profile.json"
    save(profile, data)
    (root / "corpus.jsonl").write_text('{"id":"one","text":"hello"}\n')
    (root / "requirements.txt").write_text("numpy==2.4.2\nonnxruntime==1.30.0\n")
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("test interpreter")
    identity = {
        "python": {
            "status": python_status,
            "identity": {
                "packages": packages if packages is not None else [("numpy", "2.4.2"), ("onnxruntime", "1.30.0")]
            },
        },
        "dotnet": {"status": dotnet_status},
    }
    monkeypatch.setattr(candidate, "environment_identity", lambda *args: identity)
    monkeypatch.setattr(candidate, "code_inputs", lambda *args, **kwargs: {"declared-code": Path(__file__)})
    recipe = {
        "manifest": "profile.json",
        "reference_corpus": "corpus.jsonl",
        "reference_environment": "requirements.txt",
    }
    _data, stages, shared = candidate.candidate_stages(
        recipe, repo_root=root, snapshot=snapshot, reference_python=python, threads=2, timeout=10
    )
    return root, snapshot, stages, shared, identity


@pytest.mark.parametrize("packages", [[("numpy", "2.4.2")], [("numpy", "2.4.1"), ("onnxruntime", "1.30.0")]])
def test_environment_rejects_missing_packages_and_wrong_versions(tmp_path, monkeypatch, packages):
    _root, _snapshot, stages, _shared, _identity = setup_stages(tmp_path, monkeypatch, packages=packages)
    out = tmp_path / "environment"
    with pytest.raises(PipelineError, match="differs from lock"):
        stages[1].execute(out, {})
    assert not (out / "environment.json").exists()


@pytest.mark.parametrize("tool", ["python", "dotnet"])
def test_environment_rejects_unavailable_toolchains(tmp_path, monkeypatch, tool):
    settings = {f"{tool}_status": "unavailable"}
    _root, _snapshot, stages, _shared, _identity = setup_stages(tmp_path, monkeypatch, **settings)
    with pytest.raises(PipelineError, match="must be available"):
        stages[1].execute(tmp_path / "environment", {})


def test_local_wheel_version_and_environment_identity_are_declared(tmp_path, monkeypatch):
    _root, _snapshot, stages, shared, identity = setup_stages(
        tmp_path, monkeypatch, packages=[("numpy", "2.4.2"), ("onnxruntime", "1.30.0+cpu")]
    )
    out = tmp_path / "environment"
    assert stages[1].execute(out, {})["reference_environment"] == "verified"
    assert stages[0].parameters["environment"] == identity
    assert stages[1].parameters["environment"] == identity
    assert {"candidate", "reference_requirements", "declared-code"}.issubset(shared)
    assert json.loads((out / "environment.json").read_text()) == json.loads(json.dumps(identity))


@pytest.mark.parametrize("target", ["snapshot", "protected", "source-parent"])
def test_base_stages_guard_original_and_protected_output_roots(tmp_path, monkeypatch, target):
    root, snapshot, stages, _shared, _identity = setup_stages(tmp_path, monkeypatch)
    destination = {
        "snapshot": snapshot / "new-output",
        "protected": root / "model-pipeline/protected/new-output",
        "source-parent": tmp_path,
    }[target]
    with pytest.raises(PipelineError, match="overlaps"):
        stages[0].execute(destination, {})


@pytest.mark.parametrize("contents", ["numpy>=2\n", "numpy\n", "# empty lock\n", "--index-url https://example.test\n"])
def test_requirements_reject_unpinned_or_empty_locks(tmp_path, contents):
    lock = tmp_path / "requirements.txt"
    lock.write_text(contents)
    with pytest.raises(PipelineError):
        candidate._requirements(lock)


def test_loader_rejects_duplicate_json_identity_fields(tmp_path):
    _snapshot, data = source(tmp_path)
    profile = tmp_path / "duplicate-field.json"
    profile.write_text('{"id":"first-identity",' + json.dumps(data)[1:])
    with pytest.raises(PipelineError, match="duplicate JSON field: id"):
        candidate.load_candidate(profile)


def test_source_reader_rejects_mutation_after_assets_before_callback(tmp_path):
    snapshot, data = source(tmp_path)
    asset_evidence = tmp_path / "asset-evidence"
    asset_evidence.mkdir()
    candidate.verify_assets(data, snapshot, asset_evidence)
    calls = []

    def reader(out, dependencies):
        calls.append(True)
        (out / "read.json").write_text("{}")
        return {}

    wrapped = candidate.source_reader(Stage("reference", (), {}, {}, reader), data, snapshot)
    (snapshot / "config.json").write_bytes(b"[]\n")
    output = tmp_path / "reference-evidence"
    output.mkdir()
    mismatch = r"asset mismatch: config\.json: expected \d+ bytes sha256 [0-9a-f]{64}, found 3 bytes"
    with pytest.raises(PipelineError, match=mismatch):
        wrapped.execute(output, {})
    assert calls == []
    assert wrapped.inputs["original_snapshot"] == snapshot


def test_source_reader_and_runner_reject_mutation_during_callback(tmp_path):
    snapshot, data = source(tmp_path)

    def reader(out, dependencies):
        (out / "read.json").write_text("{}")
        (snapshot / "config.json").write_bytes(b"[]\n")
        return {}

    wrapped = candidate.source_reader(Stage("reference", (), {}, {}, reader), data, snapshot)
    record = run_pipeline([wrapped], tmp_path / "run", provenance={})
    assert record["status"] == "failed"
    assert "input changed during execution" in record["stages"]["reference"]["failure"]["message"]


def test_loader_rejects_duplicate_json_fields_inside_source_lock(tmp_path):
    _snapshot, data = source(tmp_path)
    raw = json.dumps(data)
    revision = '"revision": "' + "a" * 40 + '"'
    raw = raw.replace(revision, '"revision":"' + "b" * 40 + '",' + revision)
    profile = tmp_path / "nested-duplicate-field.json"
    profile.write_text(raw)
    with pytest.raises(PipelineError, match="duplicate JSON field: revision"):
        candidate.load_candidate(profile)


@pytest.mark.parametrize(
    "contents",
    [
        "numpy==2.4.1\nNumPy==2.4.2\n",
        "numpy==2.4.2\nNumPy==2.4.2\n",
        "onnxruntime_genai==0.17.1\nonnxruntime-genai==0.17.1\n",
    ],
)
def test_requirements_reject_duplicate_normalized_package_names(tmp_path, contents):
    lock = tmp_path / "requirements.txt"
    lock.write_text(contents)
    with pytest.raises(PipelineError, match="duplicate normalized package"):
        candidate._requirements(lock)


@pytest.mark.parametrize(
    "backend,version,package", [("onnx", "1.30.0", "onnxruntime"), ("ortgenai", "0.17.1", "onnxruntime-genai")]
)
@pytest.mark.parametrize("locked_version", [None, "1.29.0"])
def test_reference_lock_must_pin_declared_runtime_package(
    tmp_path, monkeypatch, backend, version, package, locked_version
):
    root, snapshot, _stages, _shared, _identity = setup_stages(
        tmp_path, monkeypatch, packages=[("numpy", "2.4.2"), (package, "1.29.0")]
    )
    data = json.loads((root / "profile.json").read_text())
    data["runtime"].update({"backend": backend, "version": version})
    save(root / "profile.json", data)
    (root / "requirements.txt").write_text(
        "numpy==2.4.2\n" + (f"{package}=={locked_version}\n" if locked_version is not None else "")
    )
    recipe = {
        "manifest": "profile.json",
        "reference_corpus": "corpus.jsonl",
        "reference_environment": "requirements.txt",
    }
    with pytest.raises(PipelineError, match=f"reference lock must pin {package}"):
        candidate.candidate_stages(
            recipe,
            repo_root=root,
            snapshot=snapshot,
            reference_python=tmp_path / "venv/bin/python",
            threads=2,
            timeout=10,
        )
