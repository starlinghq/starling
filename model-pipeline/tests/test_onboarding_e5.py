"""Cheap orchestration checks: no model loading, inference, or SDK build."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from model_pipeline import evaluate
from model_pipeline import onboarding_e5 as e5
from model_pipeline.onboarding_runner import PipelineError

REPO = Path(__file__).resolve().parents[2]
RECIPE = {
    "recipe_version": 1,
    "id": "test-e5",
    "plugin": e5.PLUGIN,
    "manifest": f"model-pipeline/registry/{e5.PROFILE}/manifest.yaml",
    "reference_corpus": "model-pipeline/reference/corpus/tokenizer-corpus-v1.jsonl",
    "reference_environment": "model-pipeline/reference/encoder/requirements.txt",
}


@pytest.fixture
def profile(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    asset = snapshot / "config.json"
    asset.write_bytes(b"{}")
    man = copy.deepcopy(e5.manifest.load(REPO / RECIPE["manifest"]))
    man["origin"]["files"] = [{"path": asset.name, "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}]
    for key in ("manifest", "reference_corpus", "reference_environment"):
        path = root / RECIPE[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(man) if key == "manifest" else "", encoding="utf-8")
    protected = root / man["qualification"]["fixtures"]
    protected.mkdir(parents=True)
    (protected / "tolerances.yaml").write_text("{}")
    identity = {
        "python": {"status": "available", "identity": {"packages": [(p, "1") for p in e5.PACKAGES]}},
        "dotnet": {"status": "available", "stdout": "SDK 10"},
    }
    monkeypatch.setattr(e5, "_environment_identity", lambda *args: copy.deepcopy(identity))
    monkeypatch.setattr(e5.protection, "resolve_base", lambda *args: ("b" * 40, "test base"))
    monkeypatch.setattr(e5.fixtures, "git_head", lambda *args: "a" * 40 + "-dirty")
    monkeypatch.setattr(e5, "_assert_committed_code", lambda *args: None)
    return root, snapshot, identity


def stages(profile, **kwargs):
    root, snapshot, _ = profile
    return {
        s.name: s
        for s in e5.build_stages(
            RECIPE, repo_root=root, snapshot=snapshot, reference_python=Path(sys.executable), **kwargs
        )
    }


def test_plan_requires_no_toolchain_or_snapshot(monkeypatch):
    monkeypatch.setattr(e5, "_probe", lambda *args, **kwargs: pytest.fail("plan must not probe"))
    plan = e5.describe_plan(RECIPE, repo_root=REPO)
    assert plan["required_gates"] == ["G1", "G2"]
    assert "proposed" in plan["reference_status"]
    assert "incomplete" in plan["release_status"]


def test_graph_and_environment_are_cache_inputs(profile):
    first = stages(profile)
    assert list(first) == ["assets", "environment", "reference", "export", "dotnet", "evaluate", "review"]
    assert "reference" in first["export"].dependencies
    assert "snapshot" in first["assets"].inputs
    profile[2]["python"]["identity"]["packages"][0] = ("torch", "2")
    second = stages(profile)
    assert first["reference"].parameters != second["reference"].parameters
    assert first["reference"].parameters["bootstrap"] == e5.BOUNDED_RUNNER


def test_readme_only_commit_does_not_invalidate_inference(profile, monkeypatch):
    first = stages(profile, distribution=True)
    monkeypatch.setattr(e5.fixtures, "git_head", lambda *args: "c" * 40)
    second = stages(profile, distribution=True)
    for name in ("assets", "environment", "reference", "export", "dotnet", "evaluate"):
        assert first[name].parameters == second[name].parameters
    assert first["review"].parameters != second["review"].parameters
    assert first["distribution"].parameters != second["distribution"].parameters


def test_code_identity_is_scoped_and_includes_dotnet_build_settings():
    reference = e5._code_inputs(REPO, "reference")
    dotnet = e5._code_inputs(REPO, "dotnet")
    assert "Directory.Build.props" in dotnet
    assert "ai-dotnet/src/Starlings.Models.E5/E5Encoder.cs" in dotnet
    assert not any(path.startswith("ai-dotnet/") for path in reference)
    assert "model-pipeline/src/model_pipeline/onboarding_e5.py" in reference
    distribution = e5._code_inputs(REPO, "distribution")
    assert "tools/release-drill/ai-consumer/AiConsumer.csproj.in" in distribution
    assert "ai-dotnet/samples/E5Search/Program.cs" in distribution
    assert any("Models.E5.MLNet/" in path for path in distribution)


def test_source_lock_verifies_assets_and_observes_ancillary(profile, tmp_path):
    (profile[1] / "extra.json").write_text("extra")
    out = tmp_path / "output"
    metadata = stages(profile)["assets"].execute(out, {})
    lock = json.loads((out / "source-lock.json").read_text())
    assert metadata["manifest_assets_verified"] == 1
    assert lock["assets"][0]["verification"] == "manifest-size-and-sha256"
    assert lock["ancillary_assets"][0]["verification"].startswith("observed-only")
    assert lock["ancillary_assets"][0]["sha256"] == hashlib.sha256(b"extra").hexdigest()


def test_hugging_face_file_symlinks_are_verified_and_target_is_input(profile, tmp_path):
    asset = profile[1] / "config.json"
    blob = tmp_path / "blobs" / "asset"
    blob.parent.mkdir()
    asset.rename(blob)
    asset.symlink_to(blob)
    stage = stages(profile)["assets"]
    assert stage.inputs["snapshot-link:config.json"] == asset
    assert stage.execute(tmp_path / "output", {})["manifest_assets_verified"] == 1
    with pytest.raises(PipelineError, match="overlaps"):
        stage.execute(blob.parent, {})


def test_distribution_rejects_code_not_at_committed_head(monkeypatch):
    monkeypatch.setattr(e5, "_probe", lambda *args, **kwargs: {"status": "unavailable"})
    with pytest.raises(PipelineError, match="match committed HEAD"):
        e5._assert_committed_code(REPO, {REPO / "Directory.Build.props"}, 10)


@pytest.mark.parametrize("bytes_", [b"bad", b"xx"])
def test_size_or_hash_mismatch_blocks_execution(profile, tmp_path, bytes_):
    (profile[1] / "config.json").write_bytes(bytes_)
    with pytest.raises(PipelineError, match="asset mismatch"):
        stages(profile)["assets"].execute(tmp_path / "output", {})


def test_snapshot_output_overlap_is_rejected(profile):
    with pytest.raises(PipelineError, match="overlaps"):
        stages(profile)["assets"].execute(profile[1] / "output", {})
    assert not (profile[1] / "output").exists()


def test_missing_reference_dependencies_are_actionable(profile, tmp_path):
    profile[2]["python"]["identity"]["packages"] = []
    with pytest.raises(PipelineError, match="missing packages: torch"):
        stages(profile)["environment"].execute(tmp_path / "output", {})
    assert (tmp_path / "output/environment.json").is_file()


def test_runtime_only_dotnet_install_is_missing_sdk(monkeypatch):
    def probe(argv, **kwargs):
        if argv[-1] == "--version":
            return {"status": "unavailable", "error": "No SDKs were found"}
        if argv[-1] == "--info":
            return {"status": "available", "stdout": "runtime 10"}
        return {"status": "available", "stdout": '{"packages": []}'}

    monkeypatch.setattr(e5, "_probe", probe)
    identity = e5._environment_identity(Path(sys.executable), "dotnet", REPO, 5)
    assert identity["dotnet"]["status"] == "unavailable"
    assert "No SDKs" in identity["dotnet"]["error"]


def test_reference_versions_must_match_recipe_lock(profile, tmp_path):
    (profile[0] / RECIPE["reference_environment"]).write_text("torch==7 \\\n    --hash=sha256:abc\n")
    with pytest.raises(PipelineError, match="expected 7, installed 1"):
        stages(profile)["environment"].execute(tmp_path / "output", {})


def test_cpu_local_version_matches_public_version_lock(profile, tmp_path):
    (profile[0] / RECIPE["reference_environment"]).write_text("torch==2.14.1\n")
    profile[2]["python"]["identity"]["packages"][0] = ("torch", "2.14.1+cpu")
    assert stages(profile)["environment"].execute(tmp_path / "output", {})["toolchain_verified"] is True
    assert not e5._locked_version_matches("2.14.1+cuda", "2.14.1+cpu")


@pytest.mark.parametrize("threads", [0, 65, True, 1.5])
def test_invalid_thread_budget(profile, threads):
    with pytest.raises(PipelineError, match="threads"):
        stages(profile, threads=threads)


def test_unsupported_profile_fails_explicitly(profile):
    path = profile[0] / RECIPE["manifest"]
    man = yaml.safe_load(path.read_text())
    man["execution"]["precision"] = "fp16"
    path.write_text(yaml.safe_dump(man))
    with pytest.raises(PipelineError, match="unsupported profile"):
        stages(profile)


def test_provider_options_cannot_be_silently_ignored(profile):
    path = profile[0] / RECIPE["manifest"]
    man = yaml.safe_load(path.read_text())
    man["execution"]["provider_options"] = {"arena_extend_strategy": "kSameAsRequested"}
    path.write_text(yaml.safe_dump(man))
    with pytest.raises(PipelineError, match="unsupported profile"):
        stages(profile)


def test_tokenizer_package_version_must_match_manifest(profile):
    props = profile[0] / "ai-dotnet/Directory.Packages.props"
    props.parent.mkdir(parents=True, exist_ok=True)
    props.write_text(
        '<Project><ItemGroup><PackageVersion Include="Starlings.Tokenizers" Version="[9.0.0-preview.1]" />'
        '<PackageVersion Include="Microsoft.ML.OnnxRuntime" Version="[1.30.0]" />'
        '<PackageVersion Include="Microsoft.ML.OnnxRuntime.Managed" Version="[1.30.0]" />'
        "</ItemGroup></Project>"
    )
    with pytest.raises(PipelineError, match="unsupported tokenizer version"):
        stages(profile)


def test_tokenizer_package_must_be_pinned_exactly(profile):
    props = profile[0] / "ai-dotnet/Directory.Packages.props"
    props.parent.mkdir(parents=True, exist_ok=True)
    props.write_text(
        '<Project><ItemGroup><PackageVersion Include="Starlings.Tokenizers" Version="0.3.0-preview.1" />'
        '<PackageVersion Include="Microsoft.ML.OnnxRuntime" Version="[1.30.0]" />'
        '<PackageVersion Include="Microsoft.ML.OnnxRuntime.Managed" Version="[1.30.0]" />'
        "</ItemGroup></Project>"
    )
    with pytest.raises(PipelineError, match="exact version"):
        stages(profile)


@pytest.mark.parametrize(
    "keys,value",
    [
        (("behavior", "max_input_tokens"), 2048),
        (("behavior", "truncation"), "truncate-left"),
        (("behavior", "offsets_unit"), "utf16-code-units"),
        (("behavior", "defaults", "dimension"), 768),
        (("behavior", "defaults", "pooling"), "CLS"),
        (("behavior", "defaults", "normalize"), "none"),
        (("behavior", "defaults", "input_prefix"), "automatically add query: "),
        (("processing", "tokenizer", "profile_id"), "qwen-bpe"),
        (("processing", "tokenizer", "algorithm"), "bpe"),
        (("processing", "tokenizer", "backend_tokenizer_dependency"), True),
        (("execution", "backend", "version"), "999.0.0"),
        (("execution", "backend", "binding"), "other.binding"),
        (("execution", "variant"), "approximate"),
        (("execution", "graph", "outputs"), [{"name": "embedding", "type": "float32", "shape": ["batch_size", 384]}]),
        (
            ("execution", "graph", "inputs"),
            [{"name": "input_ids", "type": "int64", "shape": ["batch_size", "sequence_length"]}],
        ),
    ],
)
def test_incompatible_schema_valid_e5_contract_is_rejected(profile, keys, value):
    path = profile[0] / RECIPE["manifest"]
    man = yaml.safe_load(path.read_text())
    node = man
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    path.write_text(yaml.safe_dump(man))
    assert not e5.manifest.validate_file(path), "mutation must exercise plugin contract rather than schema validation"
    with pytest.raises(PipelineError, match="unsupported profile"):
        stages(profile)


@pytest.mark.parametrize("package", ["Microsoft.ML.OnnxRuntime", "Microsoft.ML.OnnxRuntime.Managed"])
def test_central_ort_pin_must_match_manifest(profile, package):
    props = profile[0] / "ai-dotnet/Directory.Packages.props"
    props.parent.mkdir()
    text = (REPO / "ai-dotnet/Directory.Packages.props").read_text()
    text = text.replace(f'Include="{package}" Version="[1.30.0]"', f'Include="{package}" Version="[1.31.0]"')
    props.write_text(text)
    with pytest.raises(PipelineError, match="centrally pinned"):
        stages(profile)


@pytest.mark.parametrize("package", ["Microsoft.ML.OnnxRuntime", "Microsoft.ML.OnnxRuntime.Managed"])
def test_project_ort_lock_must_match_manifest(profile, package):
    props = profile[0] / "ai-dotnet/Directory.Packages.props"
    props.parent.mkdir()
    props.write_bytes((REPO / "ai-dotnet/Directory.Packages.props").read_bytes())
    lock = (profile[0] / e5.FIDELITY).parent / "packages.lock.json"
    lock.parent.mkdir(parents=True)
    data = json.loads(((REPO / e5.FIDELITY).parent / "packages.lock.json").read_text())
    data["dependencies"]["net10.0"][package]["resolved"] = "1.31.0"
    lock.write_text(json.dumps(data))
    with pytest.raises(PipelineError, match="resolves .*expected 1.30.0"):
        stages(profile)


def test_reference_only_writes_proposals_and_commands_are_bounded(profile, tmp_path, monkeypatch):
    commands = []

    def command(argv, **kwargs):
        commands.append((argv, kwargs))
        kwargs["log_path"].write_text("okay")
        return {"returncode": 0}

    monkeypatch.setattr(e5, "run_command", command)
    graph = stages(profile, threads=3, timeout=19)
    ref = tmp_path / "ref"
    metadata = graph["reference"].execute(ref, {})
    graph["export"].execute(tmp_path / "export", {})
    graph["dotnet"].execute(tmp_path / "dotnet", {})
    assert metadata["automatically_adopted"] is False
    assert json.loads((ref / "proposal.json").read_text())["status"] == "proposed"
    assert commands[0][0][2] == "3"
    assert "--intra-op-threads" in commands[1][0]
    assert "--threads" in commands[2][0]
    build = Path(commands[2][0][commands[2][0].index("--artifacts-path") + 1])
    assert build.parent == tmp_path / "dotnet"
    assert not build.exists()
    assert "-p:RestoreLockedMode=true" in commands[2][0]
    assert all(kwargs["timeout"] == 19 and kwargs["env"]["OMP_NUM_THREADS"] == "3" for _, kwargs in commands)
    assert all(
        str(profile[0] / "model-pipeline/protected") not in argv[argv.index("--out") + 1] for argv, _ in commands
    )


def test_evaluation_requires_protected_gates_and_preserves_failure_report(profile, tmp_path, monkeypatch):
    deps = {name: tmp_path / name for name in ("export", "dotnet")}
    for directory in deps.values():
        directory.mkdir()
    (deps["export"] / "export.results.json").write_text("{}")
    (deps["export"] / "hidden.f32").write_bytes(b"abcd")
    seen = {}

    def evaluate_(path, directory, **kwargs):
        seen.update(kwargs)
        assert (directory / "hidden.f32").read_bytes() == b"abcd"
        return evaluate.Evaluation({"gates": {"G1": {"status": "passed"}, "G2": {"status": "not-run"}}}, 4)

    monkeypatch.setattr(e5.evaluate, "evaluate", evaluate_)
    monkeypatch.setattr(e5.evaluate, "write_report", lambda result, out: e5._json(out / "report.json", result.report))
    with pytest.raises(PipelineError, match="protected fidelity failed"):
        stages(profile)["evaluate"].execute(tmp_path / "evaluation", deps)
    assert seen["require_gates"] == ("G1", "G2")
    assert seen["protected_base"] == "auto"
    assert (tmp_path / "evaluation/report.json").exists()


def test_review_does_not_claim_release_qualification(profile, tmp_path):
    deps = {name: tmp_path / name for name in ("assets", "evaluate", "reference")}
    for directory in deps.values():
        directory.mkdir()
    e5._json(deps["assets"] / "source-lock.json", {"assets": []})
    report = {"gates": {g: {"status": "passed"} for g in ("G1", "G2")}, "protected_evidence": {}, "inputs": {}}
    e5._json(deps["evaluate"] / "report.json", report)
    (deps["evaluate"] / "report.md").write_text("passed")
    out = tmp_path / "review"
    stages(profile)["review"].execute(out, deps)
    data = json.loads((out / "review.json").read_text())
    assert data["release_qualification"]["status"] == "incomplete"
    assert data["reference"]["status"] == "proposed"
    assert data["distribution"]["status"] == "not-run"
    assert data["support_status_changed"] is False


def test_distribution_uses_committed_head_and_real_model(profile, tmp_path, monkeypatch):
    commands = []

    def command(argv, **kwargs):
        commands.append(argv)
        kwargs["log_path"].write_text("okay")
        if argv[0] == "git" and argv[1] == "clone":
            Path(argv[-1]).mkdir()
        if argv[0] == "bash":
            artifacts = kwargs["cwd"] / "artifacts/release-drill-ai"
            artifacts.mkdir(parents=True)
            (artifacts / "sample.log").write_text("real consumer")
        return {"returncode": 0}

    monkeypatch.setattr(e5, "run_command", command)
    graph = stages(profile, distribution=True)
    out = tmp_path / "distribution"
    metadata = graph["distribution"].execute(out, {})
    assert commands[1][-1] == "a" * 40
    assert "--seed-model-from" in commands[2]
    assert "--no-model" not in commands[2]
    assert metadata["candidate_commit"] == "a" * 40
    assert (out / "artifacts/sample.log").exists()
    assert not list(out.glob("committed-drill-*"))
