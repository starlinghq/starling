from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_pipeline import onboarding_distribution as distribution
from model_pipeline.onboarding_runner import PipelineError, Stage

TOKENIZER_HASH = base64.b64encode(hashlib.sha512(b"published tokenizer package").digest()).decode()


def versions():
    return {**{name: "0.1.0-preview.99" for name in distribution.PACKAGES}, distribution.TOKENIZER: "0.3.0-preview.99"}


def package_feed(tmp_path, *, altered=None):
    feed = tmp_path / "feed"
    feed.mkdir()
    selected = versions()
    for name in distribution.PACKAGES:
        version = selected[name]
        document = ET.Element("package")
        metadata = ET.SubElement(document, "metadata")
        ET.SubElement(metadata, "id").text = name
        ET.SubElement(metadata, "version").text = version
        ET.SubElement(metadata, "repository", commit="a" * 40)
        group = ET.SubElement(ET.SubElement(metadata, "dependencies"), "group", targetFramework="net10.0")
        for package, pin in distribution._dependencies(selected)[name].items():
            ET.SubElement(group, "dependency", id=package, version=pin)
        if altered and name == "Starlings.Models.Bert":
            altered(metadata)
        path = feed / f"{name}.{version}.nupkg"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{name}.nuspec", ET.tostring(document))
            archive.writestr(f"lib/net10.0/{name}.dll", b"synthetic assembly")
    return feed, selected


def test_inspects_all_committed_packages_and_exact_sibling_dependencies(tmp_path):
    feed, selected = package_feed(tmp_path)
    records = distribution._inspect_packages(feed, selected, "a" * 40)
    assert len(records) == len(distribution.PACKAGES) == 7
    assert distribution.TOKENIZER not in {record["id"] for record in records}
    assert all(record["source_commit"] == "a" * 40 for record in records)
    assert all(len(record["sha256"]) == 64 for record in records)
    assert records[-1]["dependencies"]["Starlings.Tokenizers"] == "[0.3.0-preview.99]"


def test_rejects_package_without_exact_sibling_pin(tmp_path):
    def loosen(metadata):
        metadata.find("dependencies/group/dependency").set("version", "0.1.0-preview.99")

    feed, selected = package_feed(tmp_path, altered=loosen)
    with pytest.raises(PipelineError, match="nuspec dependencies"):
        distribution._inspect_packages(feed, selected, "a" * 40)


def test_rejects_package_from_another_commit(tmp_path):
    feed, selected = package_feed(tmp_path)
    with pytest.raises(PipelineError, match="source commit"):
        distribution._inspect_packages(feed, selected, "b" * 40)


def graph_fixture(tmp_path):
    feed, selected = package_feed(tmp_path)
    libraries = {}
    for name in (
        "Starlings.Models.Bert",
        "Starlings.Runtime.Onnx",
        "Starlings.Acquisition",
    ):
        package = feed / f"{name}.{selected[name]}.nupkg"
        libraries[f"{name}/{selected[name]}"] = {
            "type": "package",
            "sha512": base64.b64encode(hashlib.sha512(package.read_bytes()).digest()).decode(),
        }
    # The tokenizer comes from nuget.org, not the candidate feed: its bytes are the ones the lock files record.
    tokenizer = f"{distribution.TOKENIZER}/{selected[distribution.TOKENIZER]}"
    libraries[tokenizer] = {"type": "package", "sha512": TOKENIZER_HASH}
    libraries["Microsoft.ML.OnnxRuntime/1.30.0"] = {"type": "package", "sha512": "external package"}
    path = tmp_path / "project.assets.json"
    path.write_text(json.dumps({"libraries": libraries}))
    return path, feed, selected, libraries


def test_consumer_restore_graph_verifies_packed_bytes(tmp_path):
    path, feed, selected, _libraries = graph_fixture(tmp_path)
    graph = distribution._consumer_graph(path, feed, selected, "onnx", TOKENIZER_HASH)
    assert graph["only_packages"] is True
    assert graph["packed_package_bytes_verified"] is True
    assert graph["tokenizer_package_bytes_verified"] is True


@pytest.mark.parametrize(
    "failure",
    ["project", "package-bytes", "tokenizer-bytes", "tokenizer-version", "foreign-tokenizer", "gpu", "native-version"],
)
def test_rejects_wrong_consumer_restore_graph(tmp_path, failure):
    path, feed, selected, libraries = graph_fixture(tmp_path)
    if failure == "project":
        libraries[next(iter(libraries))]["type"] = "project"
    elif failure == "package-bytes":
        libraries[next(iter(libraries))]["sha512"] = "wrong bytes"
    elif failure == "tokenizer-bytes":
        libraries[f"{distribution.TOKENIZER}/{selected[distribution.TOKENIZER]}"]["sha512"] = "other bytes"
    elif failure == "tokenizer-version":
        entry = libraries.pop(f"{distribution.TOKENIZER}/{selected[distribution.TOKENIZER]}")
        libraries[f"{distribution.TOKENIZER}/0.3.0-preview.100"] = entry
    elif failure == "foreign-tokenizer":
        libraries["Microsoft.ML.Tokenizers/2.0.0"] = {"type": "package"}
    elif failure == "gpu":
        libraries["Microsoft.ML.OnnxRuntime.Gpu/1.30.0"] = {"type": "package"}
    else:
        del libraries["Microsoft.ML.OnnxRuntime/1.30.0"]
        libraries["Microsoft.ML.OnnxRuntime/1.29.0"] = {"type": "package"}
    path.write_text(json.dumps({"libraries": libraries}))
    with pytest.raises(PipelineError):
        distribution._consumer_graph(path, feed, selected, "onnx", TOKENIZER_HASH)


def test_external_consumer_project_has_only_two_exact_package_references():
    project = ET.fromstring(
        distribution._consumer_project(
            "Starlings.Models.Bert", "0.1.0-preview.99", "Microsoft.ML.OnnxRuntime", "1.30.0"
        )
    )
    assert not list(project.iter("ProjectReference"))
    references = {item.attrib["Include"]: item.attrib["Version"] for item in project.iter("PackageReference")}
    assert references == {
        "Starlings.Models.Bert": "[0.1.0-preview.99]",
        "Microsoft.ML.OnnxRuntime": "[1.30.0]",
    }


def seed_fixture(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    files = []
    for name, contents in (
        ("onnx/model.onnx", b"graph"),
        ("tokenizer.json", b"{}"),
        ("weights.safetensors", b"original weights"),
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        files.append({"path": name, "size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()})
    candidate = {
        "id": "unit-bert",
        "origin": {"repo": "test/model", "revision": "a" * 40, "files": files},
        "behavior": {"graph_file": "onnx/model.onnx", "tokenizer_file": "tokenizer.json"},
    }
    return source, candidate


def test_bert_seed_copies_only_verified_runtime_assets(tmp_path):
    source, candidate = seed_fixture(tmp_path)
    target = tmp_path / "model"
    records = distribution._seed(candidate, source, target, bundle=False)
    assert len(records) == 2
    assert (target / "onnx/model.onnx").read_bytes() == b"graph"
    assert not (target / "weights.safetensors").exists()


def test_seed_rejects_source_corruption(tmp_path):
    source, candidate = seed_fixture(tmp_path)
    (source / "onnx/model.onnx").write_bytes(b"wrong")
    with pytest.raises(PipelineError, match="seed mismatch"):
        distribution._seed(candidate, source, tmp_path / "model", bundle=False)


def test_bundle_seed_rejects_wrong_original_identity(tmp_path):
    source, candidate = seed_fixture(tmp_path)
    (source / "bundle-lock.json").write_text(
        json.dumps({"profile": "wrong", "origin": candidate["origin"], "files": []})
    )
    with pytest.raises(PipelineError, match="another original candidate"):
        distribution._seed(candidate, source, tmp_path / "model", bundle=True)


def test_offline_fails_explicitly_when_network_namespace_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution.shutil, "which", lambda name: "/usr/bin/unshare")
    monkeypatch.setattr(distribution, "_probe", lambda *args, **kwargs: {"status": "unavailable"})
    with pytest.raises(PipelineError, match="network namespace is unavailable"):
        distribution._offline(
            "dotnet", tmp_path / "app.dll", [], cwd=tmp_path, log=tmp_path / "log", env={}, timeout=10
        )


def test_offline_uses_namespace_and_never_proxy_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution.shutil, "which", lambda name: "/usr/bin/unshare")
    monkeypatch.setattr(distribution, "_probe", lambda *args, **kwargs: {"status": "available"})
    commands = []
    monkeypatch.setattr(
        distribution, "run_command", lambda command, **kwargs: commands.append(command) or {"exit_code": 0}
    )
    distribution._offline(
        "dotnet", tmp_path / "app.dll", ["--out", "out"], cwd=tmp_path, log=tmp_path / "log", env={}, timeout=10
    )
    assert commands == [["unshare", "-rn", "dotnet", str(tmp_path / "app.dll"), "--out", "out"]]


def test_distribution_is_inserted_and_review_includes_its_result(tmp_path, monkeypatch):
    source, candidate = seed_fixture(tmp_path)
    candidate["runtime"] = {"backend": "onnx"}
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("candidate.json", "corpus.jsonl", "requirements.txt", "source.txt"):
        (root / name).write_text("test")
    monkeypatch.setattr(distribution, "load_candidate", lambda path: candidate)
    monkeypatch.setattr(distribution, "code_inputs", lambda *args, **kwargs: {"source": root / "source.txt"})
    monkeypatch.setattr(distribution, "_head", lambda *args: "a" * 40)
    monkeypatch.setattr(distribution, "_versions", lambda *args: versions())
    monkeypatch.setattr(distribution, "tokenizer_content_hash", lambda *args: TOKENIZER_HASH)

    def callback(out, deps):
        (out / "review.json").write_text('{"reference_status":"proposed"}')
        (out / "review.md").write_text("Development parity passed; distribution is not run.")
        return {"reference_status": "proposed"}

    stages = [
        Stage(name, (), {}, {}, callback) for name in ("assets", "reference", "onnx", "dotnet", "evaluate", "review")
    ]
    recipe = {
        "manifest": "candidate.json",
        "reference_corpus": "corpus.jsonl",
        "reference_environment": "requirements.txt",
    }
    assembled = distribution.append_distribution(
        stages, recipe=recipe, repo_root=root, snapshot=source, threads=2, timeout=10
    )
    assert [stage.name for stage in assembled][-2:] == ["distribution", "review"]
    assert "original_snapshot" in assembled[-2].inputs
    assert "distribution" in assembled[-1].dependencies
    report_dir = tmp_path / "distribution"
    report_dir.mkdir()
    (report_dir / "distribution.json").write_text('{"status":"passed","source_commit":"test"}')
    review = tmp_path / "review"
    review.mkdir()
    result = assembled[-1].execute(review, {"distribution": report_dir})
    assert result["distribution"] == "passed"
    assert result["release_qualification"] == "unapproved"
    assert json.loads((review / "review.json").read_text())["distribution"]["source_commit"] == "test"
    markdown = (review / "review.md").read_text()
    assert "Development distribution **passed**" in markdown
    assert "not run" not in markdown
    assert "committed source `test`" in markdown
    assert "release qualification remain **unapproved**" in markdown


def test_distribution_rejects_incomplete_plans(tmp_path):
    with pytest.raises(PipelineError, match="complete candidate stages"):
        distribution.append_distribution([], recipe={}, repo_root=tmp_path, snapshot=tmp_path, threads=2, timeout=10)


@pytest.mark.parametrize("plugin_name", ["bert-onnx-cpu-v1", "qwen-genai-cpu-v1", "e5-onnx-cpu-v1", "kev-onnx-cpu-v1"])
def test_plugin_wiring_keeps_expensive_stage_builds_separate(tmp_path, monkeypatch, plugin_name):
    from model_pipeline import onboarding_plugins

    calls = []
    stages = [object()]
    distributed = [object(), object()]

    def build(recipe, **options):
        calls.append(("build", recipe, options))
        return stages

    def append(actual_stages, **options):
        assert actual_stages is stages
        calls.append(("append", options))
        return distributed

    monkeypatch.setattr(distribution, "append_distribution", append)
    recipe = {"plugin": plugin_name}
    result = onboarding_plugins.build_stages(
        recipe,
        recipe_path=tmp_path / "recipes/candidate.json",
        plugin=SimpleNamespace(
            build_stages=build, SHARED_DISTRIBUTION=onboarding_plugins.get_plugin(recipe).SHARED_DISTRIBUTION
        ),
        repo_root=tmp_path,
        snapshot=tmp_path / "snapshot",
        reference_python=tmp_path / "python",
        threads=2,
        timeout=10,
        distribution=True,
    )
    if plugin_name == "e5-onnx-cpu-v1":
        assert result is stages
        assert calls[0][2]["distribution"] is True
        assert len(calls) == 1
    else:
        assert result is distributed
        assert calls[0][2]["distribution"] is False
        assert calls[1][1]["recipe"]["recipe_path"] == "recipes/candidate.json"
        assert "reference_python" not in calls[1][1]
        assert recipe == {"plugin": plugin_name}


def test_committed_source_guard_refuses_untracked_or_modified_inputs(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    source = root / "source.txt"
    source.write_text("committed")
    commands = (
        ["git", "init", "-q"],
        ["git", "add", "source.txt"],
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "test"],
    )
    for command in commands:
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    distribution._assert_committed_code(root, {source}, 10)
    source.write_text("modified")
    with pytest.raises(PipelineError, match="committed HEAD"):
        distribution._assert_committed_code(root, {source}, 10)
    untracked = root / "untracked.txt"
    untracked.write_text("new input")
    with pytest.raises(PipelineError, match="committed HEAD"):
        distribution._assert_committed_code(root, {untracked}, 10)


def test_consumer_runs_after_clone_and_isolated_package_caches_are_removed(tmp_path, monkeypatch):
    from model_pipeline import onboarding_bert

    snapshot, candidate = seed_fixture(tmp_path)
    candidate["runtime"] = {"backend": "onnx"}
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("candidate.json", "corpus.jsonl", "requirements.txt", "source.txt", "global.json"):
        (root / name).write_text("test")
    program = root / distribution.TOOLS["onnx"]
    program.parent.mkdir(parents=True)
    program.write_text("committed consumer program")
    tool_main = root / distribution.TOOL_MAIN
    tool_main.parent.mkdir(parents=True)
    tool_main.write_text("committed tool entry point")
    monkeypatch.setattr(distribution, "load_candidate", lambda path: candidate)
    monkeypatch.setattr(distribution, "code_inputs", lambda *args, **kwargs: {"source": root / "source.txt"})
    monkeypatch.setattr(distribution, "_head", lambda *args: "a" * 40)
    monkeypatch.setattr(distribution, "_versions", lambda *args: versions())
    monkeypatch.setattr(distribution, "tokenizer_content_hash", lambda *args: TOKENIZER_HASH)
    monkeypatch.setattr(distribution, "_assert_committed_code", lambda *args: None)
    monkeypatch.setattr(
        distribution, "_inspect_packages", lambda *args: [{"id": "synthetic", "source_commit": "a" * 40}]
    )
    monkeypatch.setattr(distribution, "_consumer_graph", lambda *args: {"only_packages": True})
    monkeypatch.setattr(onboarding_bert, "evaluate_development", lambda *args: {"status": "passed"})
    commands = []

    def command(argv, *, cwd, log_path, timeout, env=None):
        commands.append(argv)
        log_path.write_text("synthetic bounded command log")
        if argv[:2] == ["git", "clone"]:
            shutil.copytree(root, Path(argv[-1]))
        elif len(argv) > 1 and argv[1] == "pack":
            feed = Path(argv[argv.index("-o") + 1])
            (feed / (Path(argv[2]).stem + ".nupkg")).write_bytes(b"synthetic package")
        elif len(argv) > 1 and argv[1] == "restore":
            for key in ("NUGET_PACKAGES", "NUGET_HTTP_CACHE_PATH"):
                Path(env[key]).mkdir(exist_ok=True)
            (cwd / "obj").mkdir(exist_ok=True)
            (cwd / "obj/project.assets.json").write_text("{}")
            (cwd / "packages.lock.json").write_text("{}")
        elif len(argv) > 1 and argv[1] == "publish":
            app = Path(argv[argv.index("-o") + 1])
            native = app / "runtimes/linux-x64/native"
            native.mkdir(parents=True)
            (native / "libonnxruntime.so").write_bytes(b"native")
            (app / "Consumer.dll").write_bytes(b"app")
            (app / "Consumer.deps.json").write_text("{}")
        return {"exit_code": 0}

    def offline(dotnet, app, argv, *, cwd, log, env, timeout):
        assert not (cwd.parent / "source").exists()
        assert not Path(env["NUGET_PACKAGES"]).exists()
        assert not Path(env["NUGET_HTTP_CACHE_PATH"]).exists()
        assert (cwd / "Program.cs").read_text() == "committed consumer program"
        assert (cwd / "ToolMain.cs").read_text() == "committed tool entry point"
        assert not list(ET.parse(cwd / "Consumer.csproj").iter("ProjectReference"))
        evidence = Path(argv[argv.index("--out") + 1])
        (evidence / "dotnet.json").write_text(
            '{"environment":{"onnxruntime":"1.30.0","provider":"CPUExecutionProvider"}}'
        )
        log.write_text("network-isolated synthetic consumer run")
        return {"exit_code": 0}

    monkeypatch.setattr(distribution, "run_command", command)
    monkeypatch.setattr(distribution, "_offline", offline)

    def callback(out, deps):
        return {}

    stages = [
        Stage(name, (), {}, {}, callback) for name in ("assets", "reference", "onnx", "dotnet", "evaluate", "review")
    ]
    recipe = {
        "manifest": "candidate.json",
        "reference_corpus": "corpus.jsonl",
        "reference_environment": "requirements.txt",
    }
    assembled = distribution.append_distribution(
        stages, recipe=recipe, repo_root=root, snapshot=snapshot, threads=2, timeout=10
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "reference.json").write_text("{}")
    out = tmp_path / "evidence"
    out.mkdir()
    result = assembled[-2].execute(out, {"reference": reference, "onnx": tmp_path / "onnx"})
    assert result["status"] == "passed"
    assert result["checkout_and_package_caches_removed_before_execution"] is True
    assert len([argv for argv in commands if len(argv) > 1 and argv[1] == "pack"]) == len(distribution.PACKAGES)
    assert all(
        "-m:1" in argv and "-p:RestoreLockedMode=true" in argv
        for argv in commands
        if len(argv) > 1 and argv[1] == "pack"
    )
    assert (out / "consumer.packages.lock.json").is_file()


def test_every_registered_plugin_declares_its_distribution_source():
    from model_pipeline import onboarding_plugins

    declared = {
        name: onboarding_plugins.uses_shared_distribution(onboarding_plugins.get_plugin({"plugin": name}))
        for name in onboarding_plugins.PLUGINS
    }
    assert declared == {
        "e5-onnx-cpu-v1": False,
        "bert-onnx-cpu-v1": True,
        "qwen-genai-cpu-v1": True,
        "kev-onnx-cpu-v1": True,
        "whisper-onnx-cpu-v1": True,
    }
    with pytest.raises(PipelineError, match="SHARED_DISTRIBUTION"):
        onboarding_plugins.uses_shared_distribution(SimpleNamespace(SHARED_DISTRIBUTION="yes"))
