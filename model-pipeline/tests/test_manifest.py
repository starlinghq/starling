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


def test_named_owner_is_not_a_manifest_field(tmp_path):
    path = write_variant(tmp_path, lambda m: m["maintenance"].update(owner="Someone"))
    assert any("owner" in p.message for p in validate_file(path))


def test_unknown_capability_status_is_rejected(tmp_path):
    path = write_variant(tmp_path, lambda m: m["capabilities"].update(batch="probably"))
    assert validate_file(path)


@pytest.mark.parametrize("revision", ["main", "v1.0", "0123abc"])
def test_mutable_or_short_revisions_are_rejected(tmp_path, revision):
    path = write_variant(tmp_path, lambda m: m["origin"]["weights"].update(revision=revision))
    assert validate_file(path)


DICTIONARY = {
    "source": {"path": "inference.yml", "sha256": "0" * 64, "size": 1},
    "entries": 2,
    "sha256": "1" * 64,
    "blank": 0,
    "space": True,
}


def ocr(m):
    m["behavior"]["tasks"] = ["ocr"]
    del m["processing"]["tokenizer"]
    m["processing"]["dictionary"] = dict(DICTIONARY)


def test_an_ocr_profile_states_a_dictionary_instead_of_a_tokenizer(tmp_path):
    assert validate_file(write_variant(tmp_path, ocr)) == []


def test_an_ocr_profile_needs_its_dictionary_and_no_tokenizer(tmp_path):
    def no_dictionary(m):
        ocr(m)
        del m["processing"]["dictionary"]

    def tokenizer_too(m):
        tokenizer = m["processing"]["tokenizer"]
        ocr(m)
        m["processing"]["tokenizer"] = tokenizer

    assert any("dictionary" in p.message for p in validate_file(write_variant(tmp_path, no_dictionary)))
    assert validate_file(write_variant(tmp_path, tokenizer_too))


def test_other_profiles_need_a_tokenizer_and_no_dictionary(tmp_path):
    def no_tokenizer(m):
        del m["processing"]["tokenizer"]

    def dictionary_too(m):
        m["processing"]["dictionary"] = dict(DICTIONARY)

    assert any("tokenizer" in p.message for p in validate_file(write_variant(tmp_path, no_tokenizer)))
    assert validate_file(write_variant(tmp_path, dictionary_too))


def generator(m):
    m["behavior"]["tasks"] = ["generation"]
    m["behavior"]["generation"] = {
        "tool_call_mode": "template",
        "structured_output": "untested",
        "state_reuse": "none",
        "caching_style": "none",
        "max_output_tokens": m["behavior"]["max_input_tokens"],
    }
    m["capabilities"].update(tool_calling="experimental", conversation_state_reuse="unsupported")
    m["capabilities"]["max_new_tokens"] = "experimental"


def test_generation_fields_that_follow_their_capability_rows_are_valid(tmp_path):
    assert validate_file(write_variant(tmp_path, generator)) == []


def test_generation_section_is_required_exactly_for_generation_profiles(tmp_path):
    def drop(m):
        generator(m)
        del m["behavior"]["generation"]

    def extra(m):
        tasks = m["behavior"]["tasks"]
        generator(m)
        m["behavior"]["tasks"] = tasks

    assert any("generation" in p.message for p in validate_file(write_variant(tmp_path, drop)))
    assert [p.location for p in validate_file(write_variant(tmp_path, extra))] == ["behavior"]


@pytest.mark.parametrize(
    ("capability", "status", "field", "value"),
    [
        ("tool_calling", "unsupported", "tool_call_mode", "template"),  # unsupported is none
        ("tool_calling", "untested", "tool_call_mode", "template"),  # untested is untested
        ("tool_calling", "experimental", "tool_call_mode", "none"),
        ("structured_output", "unsupported", "structured_output", "grammar"),
        ("guided_decoding", "unsupported", "structured_output", "grammar"),
        (None, None, "structured_output", "grammar"),  # no row: never invented
        (None, None, "structured_output", "none"),
        ("conversation_state_reuse", "experimental", "state_reuse", "none"),  # no value for it yet
        ("conversation_state_reuse", "untested", "caching_style", "none"),
        ("max_new_tokens", "untested", "max_output_tokens", 512),
        ("max_new_tokens", "experimental", "max_output_tokens", 8),  # not max_input_tokens
    ],
)
def test_generation_fields_must_follow_their_capability_rows(tmp_path, capability, status, field, value):
    def edit(m):
        generator(m)
        if capability:
            m["capabilities"][capability] = status
        m["behavior"]["generation"][field] = value

    problems = validate_file(write_variant(tmp_path, edit))
    assert any(p.location == f"behavior/generation/{field}" for p in problems), problems


def test_structured_output_without_a_row_is_untested_and_lfm3_style_rows_are_read(tmp_path):
    def edit(m):
        generator(m)
        m["capabilities"].pop("conversation_state_reuse")
        m["capabilities"].update(prefix_reuse="unsupported", guided_decoding="unsupported")
        m["capabilities"].pop("max_new_tokens")
        m["behavior"]["generation"].update(structured_output="none", max_output_tokens="untested")

    assert validate_file(write_variant(tmp_path, edit)) == []


def test_stable_refuses_an_untested_generation_field(tmp_path):
    def promote(m):
        generator(m)
        m["maturity"] = "stable"
        m["capabilities"] = {k: "supported" for k in m["capabilities"]}
        m["capabilities"]["conversation_state_reuse"] = "unsupported"
        m["behavior"]["generation"]["max_output_tokens"] = m["behavior"]["max_input_tokens"]
        for gate in GATES:
            m["qualification"]["gates"][gate] = {"status": "passed", "evidence": "docs/evidence/g.md"}

    problems = validate_file(write_variant(tmp_path, promote))
    assert any("generation.structured_output" in p.message for p in problems), problems


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
