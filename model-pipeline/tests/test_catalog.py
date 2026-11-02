import json
from pathlib import Path

import pytest
import yaml

from model_pipeline import catalog
from model_pipeline.cli import main
from model_pipeline.manifest import GATES, load, stable_blockers

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"


def add_profile(registry: Path, profile_id: str, edit=None, *, synthetic: bool = False) -> dict:
    manifest = load(SYNTHETIC)
    manifest["id"] = profile_id
    if not synthetic:
        manifest.pop("synthetic")
    if edit:
        edit(manifest)
    path = registry / profile_id / "manifest.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docs" / "evidence").mkdir(parents=True)
    (tmp_path / "docs" / "evidence" / "g0.md").write_text("evidence\n", encoding="utf-8")
    (tmp_path / "registry").mkdir()
    return tmp_path


def run(repo: Path, *extra: str) -> int:
    return main(
        ["catalog", str(repo / "registry"), "--out", str(repo / "docs" / "catalog"), "--repo-root", str(repo), *extra]
    )


def profile(cat: dict, profile_id: str) -> dict:
    return next(p for p in cat["profiles"] if p["id"] == profile_id)


def test_missing_capability_is_unrecorded_never_supported(repo):
    add_profile(repo / "registry", "fam/a", lambda m: m["capabilities"].update(vision="supported"))
    add_profile(repo / "registry", "fam/b")
    cat = catalog.build(repo / "registry", repo)
    assert profile(cat, "fam/a")["capabilities"]["vision"] == "supported"
    assert profile(cat, "fam/b")["capabilities"]["vision"] == "unrecorded"
    readme = catalog.render(cat, repo / "docs" / "catalog", repo)["README.md"]
    assert "| vision | supported | unrecorded |" in readme


def test_capability_tables_cover_every_recorded_capability(repo):
    add_profile(repo / "registry", "fam/a", lambda m: m["capabilities"].update(vision="experimental"))
    cat = catalog.build(repo / "registry", repo)
    readme = catalog.render(cat, repo / "docs" / "catalog", repo)["README.md"]
    for name, status in profile(cat, "fam/a")["capabilities"].items():
        assert f"| {name} | {status} |" in readme


def test_missing_gates_render_as_none_and_evidence_is_linked(repo):
    def edit(m):
        m["qualification"]["gates"] = {
            "G0": {"status": "passed", "evidence": "docs/evidence/g0.md", "date": "2026-10-01"},
            "G1": {"status": "checklist", "evidence": "docs/evidence/missing.md"},
        }

    add_profile(repo / "registry", "fam/a", edit)
    cat = catalog.build(repo / "registry", repo)
    gates = profile(cat, "fam/a")["gates"]
    assert gates["G0"] == {
        "status": "passed",
        "evidence": "docs/evidence/g0.md",
        "evidence_found": True,
        "date": "2026-10-01",
    }
    assert gates["G1"]["evidence_found"] is False
    assert all(gates[g] == {"status": "none"} for g in GATES[2:])
    readme = catalog.render(cat, repo / "docs" / "catalog", repo)["README.md"]
    assert "[passed](../evidence/g0.md)" in readme
    assert "[checklist](../evidence/missing.md) (evidence not found)" in readme


def test_compatibility_matrix_marks_unrecorded_platforms(repo):
    add_profile(repo / "registry", "fam/cpu")

    def cuda(m):
        m["compatibility"] = [{"os": "windows", "arch": "x64", "provider": "cuda", "status": "untested"}]

    add_profile(repo / "registry", "fam/cuda", cuda)
    cat = catalog.build(repo / "registry", repo)
    assert cat["platforms"] == ["linux-x64 cpu", "windows-x64 cuda"]
    assert profile(cat, "fam/cpu")["compatibility"]["windows-x64 cuda"] == {"status": "unrecorded"}
    matrix = catalog.render(cat)["compatibility-matrix.md"]
    assert "| fam/cpu | 1.0.0 | preview | supported | unrecorded |" in matrix
    assert "| fam/cuda | 1.0.0 | preview | unrecorded | untested |" in matrix


def test_stable_readiness_reuses_the_validator_rules(repo):
    manifest = add_profile(repo / "registry", "fam/a")
    cat = catalog.build(repo / "registry", repo)
    readiness = profile(cat, "fam/a")["stable_readiness"]
    blockers = stable_blockers(manifest)
    assert readiness == {
        "ready": False,
        "gates_not_passed": list(blockers.gates_not_passed),
        "untested_capabilities": list(blockers.untested_capabilities),
    }
    assert readiness["gates_not_passed"] == ["G3", "G4", "G5", "G7"]
    assert readiness["untested_capabilities"] == ["aot", "extensions_ai", "mlnet"]
    assert catalog.stable_readiness_text(readiness) == (
        "gates not passed: G3, G4, G5, G7; untested: aot, extensions_ai, mlnet"
    )


def test_profile_meeting_stable_rules_is_ready(repo):
    def ready(m):
        m["qualification"]["gates"] = {g: {"status": "passed"} for g in GATES}
        m["capabilities"] = {k: "supported" for k in m["capabilities"]}

    add_profile(repo / "registry", "fam/a", ready)
    readiness = profile(catalog.build(repo / "registry", repo), "fam/a")["stable_readiness"]
    assert readiness == {"ready": True, "gates_not_passed": [], "untested_capabilities": []}
    assert catalog.stable_readiness_text(readiness) == "meets the validator's Stable rules"


def _ready(m):
    m["qualification"]["gates"] = {g: {"status": "passed"} for g in GATES}
    m["capabilities"] = {k: "supported" for k in m["capabilities"]}


def _cuda_only(m):
    for row in m["compatibility"]:
        row["provider"] = "cuda"


def test_stable_profile_is_labelled_stable(repo):
    def stable(m):
        _ready(m)
        m["maturity"] = "stable"

    add_profile(repo / "registry", "fam/a", stable)
    p = profile(catalog.build(repo / "registry", repo), "fam/a")
    text = catalog.stable_readiness_text(
        p["stable_readiness"], maturity=p["maturity"], scheduled_regression=p["scheduled_regression"]
    )
    assert text == "stable: meets the validator's Stable rules"


def test_gpu_only_profile_awaits_scheduled_regression(repo):
    add_profile(repo / "registry", "fam/cuda", lambda m: (_ready(m), _cuda_only(m)))
    add_profile(
        repo / "registry",
        "fam/cuda-g4",
        lambda m: (_ready(m), _cuda_only(m), m["qualification"]["gates"]["G4"].update(status="failed")),
    )
    cat = catalog.build(repo / "registry", repo)
    ready, failed = profile(cat, "fam/cuda"), profile(cat, "fam/cuda-g4")
    assert not ready["scheduled_regression"]
    assert catalog.stable_readiness_text(ready["stable_readiness"], scheduled_regression=False) == (
        "all eight gates pass where recorded; promotion awaits a green per-release GPU regression run"
    )
    assert catalog.stable_readiness_text(failed["stable_readiness"], scheduled_regression=False) == (
        "gates not passed: G4; promotion also awaits a green per-release GPU regression run"
    )
    readme = catalog.render(cat, repo / "docs" / "catalog", repo)["README.md"]
    assert "promotion awaits a green per-release GPU regression run" in readme


def test_package_readme_maturity_blocks_are_generated_and_checked(repo):
    add_profile(repo / "registry", "fam/a-cpu")
    add_profile(repo / "registry", "fam/a-cuda", _cuda_only)
    add_profile(repo / "registry", "other/b")
    pkg = repo / "ai-dotnet" / "src" / "Pkg"
    pkg.mkdir(parents=True)
    readme = pkg / "README.md"
    readme.write_text(
        "# Pkg\n\n<!-- catalog:maturity profiles=fam/* -->\nhand-written\n<!-- /catalog:maturity -->\n\nRest.\n",
        encoding="utf-8",
    )
    (repo / "ai-dotnet" / "src" / "NoBlock").mkdir()
    (repo / "ai-dotnet" / "src" / "NoBlock" / "README.md").write_text("# No block\n", encoding="utf-8")
    assert run(repo, "--check") == 1
    assert run(repo) == 0
    text = readme.read_text(encoding="utf-8")
    assert "hand-written" not in text
    assert "| `fam/a-cpu` | preview | linux-x64 cpu: supported |" in text
    assert "| `fam/a-cuda` | preview | linux-x64 cuda: supported |" in text
    assert "other/b" not in text
    assert text.endswith("<!-- /catalog:maturity -->\n\nRest.\n")
    assert run(repo, "--check") == 0

    readme.write_text(
        text.replace("<!-- catalog:maturity profiles=fam/* -->", "<!-- catalog:maturity profiles=none/* -->")
    )
    assert run(repo, "--check") == 2


def test_synthetic_manifests_are_not_published(repo):
    add_profile(repo / "registry", "fam/real")
    add_profile(repo / "registry", "fam/fake", synthetic=True)
    assert [p["id"] for p in catalog.build(repo / "registry", repo)["profiles"]] == ["fam/real"]


def test_invalid_or_empty_registry_is_refused(repo):
    assert run(repo) == 2
    add_profile(repo / "registry", "fam/a", lambda m: m["capabilities"].pop("streaming"))
    assert run(repo) == 2
    assert not (repo / "docs" / "catalog").exists()


def test_check_detects_stale_and_missing_output(repo):
    add_profile(repo / "registry", "fam/a")
    out = repo / "docs" / "catalog"
    assert run(repo, "--check") == 1
    assert run(repo) == 0
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "catalog.json", "compatibility-matrix.md"]
    assert run(repo, "--check") == 0
    json.loads((out / "catalog.json").read_text(encoding="utf-8"))

    path = repo / "registry" / "fam" / "a" / "manifest.yaml"
    manifest = load(path)
    manifest["capabilities"]["aot"] = "unsupported"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    assert run(repo, "--check") == 1
    assert run(repo) == 0
    assert run(repo, "--check") == 0

    (out / "compatibility-matrix.md").unlink()
    assert run(repo, "--check") == 1


def test_output_is_deterministic(repo):
    add_profile(repo / "registry", "fam/b")
    add_profile(repo / "registry", "fam/a")
    first = catalog.generate(repo / "registry", repo / "docs" / "catalog", repo)
    assert first == catalog.generate(repo / "registry", repo / "docs" / "catalog", repo)
    assert [p["id"] for p in json.loads(first["catalog.json"])["profiles"]] == ["fam/a", "fam/b"]


def test_committed_catalog_is_up_to_date():
    built = catalog.build()
    stale = catalog.stale(catalog.render(built), catalog.DEFAULT_OUT)
    stale += [str(path) for path in catalog.stale_readmes(catalog.package_readmes(built))]
    assert not stale, f"docs/catalog is stale ({', '.join(stale)}); regenerate with: model-pipeline catalog"


def test_catalog_json_carries_the_capability_record_contract_fields(repo):
    def edit(m):
        m["behavior"]["defaults"] = {"dimension": 384, "stop_token_ids": [7, 9]}
        m["compatibility"].append({"os": "linux", "arch": "x64", "provider": "cuda", "status": "experimental"})

    add_profile(repo / "registry", "fam/a", edit)
    entry = profile(catalog.build(repo / "registry", repo), "fam/a")
    assert entry["behavior"] == {
        "max_input_tokens": 512,
        "truncation": "truncate-right",
        "dimension": 384,
        "stop_token_ids": [7, 9],
    }
    # Only providers with a supported row are qualified.
    assert entry["qualified_providers"] == ["cpu"]


def test_csharp_capability_records_are_generated_and_checked(repo):
    def edit(m):
        m["behavior"]["defaults"] = {"dimension": 384}
        m["compatibility"][0]["verified"] = "2026-10-15"
        m["capabilities"]["vision"] = "untested"

    add_profile(repo / "registry", "fam/a-cpu", edit)
    add_profile(repo / "registry", "other/b")
    pkg = repo / "ai-dotnet" / "src" / "Starlings.Models.Fam"
    pkg.mkdir(parents=True)
    source = pkg / "ProfileCapabilities.g.cs"
    source.write_text("// catalog:capabilities profiles=fam/*\n", encoding="utf-8")
    other = repo / "ai-dotnet" / "src" / "Other"
    other.mkdir()
    (other / "ProfileCapabilities.g.cs").write_text("// no header\n", encoding="utf-8")

    assert run(repo, "--check") == 1
    assert run(repo) == 0
    text = source.read_text(encoding="utf-8")
    assert "// catalog:capabilities profiles=fam/*\n" in text
    assert "namespace Starlings.Models.Fam;" in text
    assert '["fam/a-cpu"] = new ModelProfileCapabilities' in text
    assert "Maturity = ProfileMaturity.Preview," in text
    assert "MaxInputTokens = 512," in text
    assert "Truncation = TruncationPolicy.TruncatesRight," in text
    assert "Dimension = 384," in text
    assert 'QualifiedProviders = ["cpu"],' in text
    assert 'new ProfilePlatform("linux", "x64", "cpu", Status.Supported, Verified: new DateOnly(2026, 10, 15)),' in text
    assert '["vision"] = Status.Untested,' in text
    assert "other/b" not in text
    assert (other / "ProfileCapabilities.g.cs").read_text(encoding="utf-8") == "// no header\n"
    assert run(repo, "--check") == 0

    source.write_text(text.replace('"fam/a-cpu"', '"fam/x"', 1), encoding="utf-8")
    assert run(repo, "--check") == 1

    source.write_text("// catalog:capabilities profiles=none/*\n", encoding="utf-8")
    assert run(repo, "--check") == 2


def test_csharp_capability_records_use_the_project_root_namespace(repo):
    add_profile(repo / "registry", "fam/a")
    pkg = repo / "ai-dotnet" / "src" / "Starlings.Models.Fam.Onnx"
    pkg.mkdir(parents=True)
    (pkg / "Starlings.Models.Fam.Onnx.csproj").write_text(
        "<Project><PropertyGroup><!-- <RootNamespace>Old.Name</RootNamespace> -->"
        "<RootNamespace Condition=\"'$(X)' == 'y'\">Other.Name</RootNamespace>"
        "<RootNamespace>Starlings.Models.Fam</RootNamespace></PropertyGroup></Project>\n",
        encoding="utf-8",
    )
    # Read after the project named for the directory, so its namespace wins.
    (pkg / "Aaa.csproj").write_text(
        "<Project><PropertyGroup><RootNamespace>Wrong.Name</RootNamespace></PropertyGroup></Project>\n",
        encoding="utf-8",
    )
    source = pkg / "ProfileCapabilities.g.cs"
    source.write_text("// catalog:capabilities profiles=fam/*\n", encoding="utf-8")

    assert run(repo) == 0
    assert "namespace Starlings.Models.Fam;" in source.read_text(encoding="utf-8")


def test_committed_capability_sources_are_up_to_date():
    sources = catalog.capability_sources()
    assert sources, "no package opts in to a generated capability record"
    stale = catalog.stale_readmes(sources)
    assert not stale, (
        f"capability sources are stale ({', '.join(map(str, stale))}); regenerate with: model-pipeline catalog"
    )
