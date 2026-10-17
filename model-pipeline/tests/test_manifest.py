from pathlib import Path

import pytest
import yaml

from model_pipeline.cli import main
from model_pipeline.manifest import GATES, find_manifests, load, validate_file

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"


def write_variant(tmp_path: Path, edit) -> Path:
    manifest = load(SYNTHETIC)
    edit(manifest)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_synthetic_manifest_is_valid():
    assert validate_file(SYNTHETIC) == []


def test_missing_capability_is_rejected(tmp_path):
    path = write_variant(tmp_path, lambda m: m["capabilities"].pop("streaming"))
    assert any("streaming" in p.message for p in validate_file(path))


def test_unknown_capability_status_is_rejected(tmp_path):
    path = write_variant(tmp_path, lambda m: m["capabilities"].update(batch="probably"))
    assert validate_file(path)


@pytest.mark.parametrize("revision", ["main", "v1.0", "0123abc"])
def test_mutable_or_short_revisions_are_rejected(tmp_path, revision):
    path = write_variant(tmp_path, lambda m: m["origin"]["weights"].update(revision=revision))
    assert validate_file(path)


def test_stable_requires_every_gate_passed(tmp_path):
    def promote(m):
        m["maturity"] = "stable"
        m["capabilities"] = {k: "supported" for k in m["capabilities"]}

    problems = validate_file(write_variant(tmp_path, promote))
    assert [p.location for p in problems] == ["qualification/gates"]
    assert "G3" in problems[0].message


def test_stable_rejects_untested_capabilities(tmp_path):
    def promote(m):
        m["maturity"] = "stable"
        m["qualification"]["gates"] = {f"G{i}": {"status": "passed"} for i in range(8)}

    problems = validate_file(write_variant(tmp_path, promote))
    assert [p.location for p in problems] == ["capabilities"]


def test_cli_exit_codes(tmp_path):
    assert main(["validate", str(SYNTHETIC)]) == 0
    bad = write_variant(tmp_path, lambda m: m.pop("rights"))
    assert main(["validate", str(bad)]) == 1


REGISTRY = Path(__file__).resolve().parents[1] / "registry"


@pytest.mark.parametrize("path", find_manifests(REGISTRY), ids=lambda p: str(p.parent.relative_to(REGISTRY)))
def test_registry_manifests_record_every_gate(path):
    """Every registered profile states G0 to G7 explicitly; a missing gate never means passed or not applicable."""
    gates = load(path)["qualification"]["gates"]
    assert sorted(gates) == list(GATES)
    for gate, entry in gates.items():
        if entry["status"] != "not-run":
            assert entry.get("evidence"), gate
            assert entry.get("date"), gate
            assert (REGISTRY.parents[1] / entry["evidence"]).exists(), entry["evidence"]


ROOST = {
    "kernel_manifest": {"revision": "a" * 40},
    "tf32_policy": "disabled",
    "kv_dtype": "bf16",
    "attention_backend": "flashinfer",
    "sm": ["sm_86"],
    "min_driver": "580.65",
    "min_toolkit": "13.0",
}


def make_roost(m, **extra):
    m["execution"]["backend"].update(name="roost", version="0.1.0")
    m["execution"]["roost"] = {**ROOST, **extra}


def test_roost_profile_is_valid(tmp_path):
    assert validate_file(write_variant(tmp_path, make_roost)) == []


def test_roost_optional_fields_are_valid(tmp_path):
    def edit(m):
        make_roost(
            m,
            kernel_variants=[{"op": "gemm", "variant": "cublaslt", "digest": "b" * 64}],
            mesh={"ranks": 2, "tp": 2},
            layer_precision=[{"layer": 0, "weight_dtype": "int4", "kv_dtype": "fp8_e4m3", "evidence": "calib.json"}],
        )

    assert validate_file(write_variant(tmp_path, edit)) == []


def test_roost_backend_requires_roost_block(tmp_path):
    path = write_variant(tmp_path, lambda m: m["execution"]["backend"].update(name="roost", version="0.1.0"))
    assert validate_file(path)


def test_roost_block_rejected_for_other_backends(tmp_path):
    path = write_variant(tmp_path, lambda m: m["execution"].update(roost=dict(ROOST)))
    assert validate_file(path)


@pytest.mark.parametrize(
    "field,value",
    [("tf32_policy", "maybe"), ("kv_dtype", "fp4"), ("sm", ["86"]), ("min_driver", "latest"), ("mesh", {"tp": 2})],
)
def test_roost_bad_values_are_rejected(tmp_path, field, value):
    assert validate_file(write_variant(tmp_path, lambda m: make_roost(m, **{field: value})))


def test_roost_required_field_is_enforced(tmp_path):
    def edit(m):
        make_roost(m)
        del m["execution"]["roost"]["kv_dtype"]

    assert validate_file(write_variant(tmp_path, edit))
