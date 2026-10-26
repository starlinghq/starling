from pathlib import Path

import pytest
import yaml

from model_pipeline import evidence_index
from model_pipeline.cli import main
from model_pipeline.manifest import load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"


def add_profile(registry: Path, profile_id: str, gates: dict, comment: str = "") -> None:
    manifest = load(SYNTHETIC)
    manifest["id"] = profile_id
    manifest.pop("synthetic")
    manifest["qualification"]["gates"] = gates
    path = registry / profile_id / "manifest.yaml"
    path.parent.mkdir(parents=True)
    path.write_text((f"# {comment}\n" if comment else "") + yaml.safe_dump(manifest), encoding="utf-8")


def add_evidence(evidence: Path, name: str, title: str | None) -> None:
    (evidence / name).mkdir(parents=True)
    if title is not None:
        (evidence / name / "README.md").write_text(f"Intro\n\n# {title}\n\nBody\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    evidence = tmp_path / "docs" / "evidence"
    add_evidence(evidence, "gates-2026-10-14", "Gate run")
    add_evidence(evidence, "late-run", "A run (2026-10-20)")
    add_evidence(evidence, "spike", "Spike | notes")
    add_evidence(evidence, "no-readme", None)
    registry = tmp_path / "registry"
    add_profile(
        registry,
        "fam/a",
        {
            "G0": {"status": "passed", "evidence": "docs/evidence/gates-2026-10-14/README.md", "date": "2026-10-14"},
            "G1": {"status": "checklist", "evidence": "docs/evidence/gates-2026-10-14/a.md"},
            "G2": {"status": "passed", "evidence": "docs/evidence/late-run/"},
        },
    )
    add_profile(registry, "fam/b", {"G0": {"status": "not-run"}}, comment="see docs/evidence/spike/README.md")
    return tmp_path


def run(repo: Path, *extra: str) -> int:
    return main(
        ["evidence-index", "--evidence", str(repo / "docs/evidence"), "--registry", str(repo / "registry"), *extra]
    )


def test_rows_carry_date_title_gates_and_citations(repo):
    rows = {e.name: e for e in evidence_index.build(repo / "docs/evidence", repo / "registry")}
    assert rows["gates-2026-10-14"].date == "2026-10-14"
    assert rows["gates-2026-10-14"].gates == {"fam/a": ["G0", "G1 (checklist)"]}
    assert rows["late-run"].date == "2026-10-20"
    assert rows["late-run"].gates == {"fam/a": ["G2"]}
    assert rows["spike"].date == evidence_index.UNDATED
    assert rows["spike"].gates == {}
    assert rows["spike"].cited_by == {"fam/b"}
    assert rows["no-readme"].title == ""


def test_order_is_newest_first_then_undated_by_name(repo):
    names = [e.name for e in evidence_index.build(repo / "docs/evidence", repo / "registry")]
    assert names == ["late-run", "gates-2026-10-14", "no-readme", "spike"]


def test_render_escapes_cells_and_links_entries(repo):
    text = evidence_index.generate(repo / "docs/evidence", repo / "registry")
    assert (
        "| 2026-10-14 | [`gates-2026-10-14`](gates-2026-10-14/README.md) | Gate run | `fam/a`: G0, G1 (checklist) |"
        in text
    )
    assert "Spike \\| notes" in text
    assert "[`no-readme`](no-readme/) | (no title) | none | none |" in text
    assert "4 entries: 2 dated, 2 undated; 2 hold gate evidence." in text


def test_check_fails_until_written_then_passes(repo, capsys):
    assert run(repo, "--check") == 1
    assert "STALE" in capsys.readouterr().err
    assert run(repo) == 0
    assert run(repo, "--check") == 0
    add_evidence(repo / "docs/evidence", "new-2026-10-25", "New")
    assert run(repo, "--check") == 1


def test_index_itself_is_not_an_entry(repo):
    assert run(repo) == 0
    names = [e.name for e in evidence_index.build(repo / "docs/evidence", repo / "registry")]
    assert evidence_index.INDEX not in names


def test_missing_evidence_directory_is_an_error(tmp_path, repo):
    assert main(["evidence-index", "--evidence", str(tmp_path / "nope"), "--registry", str(repo / "registry")]) == 2


def test_committed_index_is_current():
    assert not evidence_index.stale(evidence_index.generate()), "regenerate with: model-pipeline evidence-index"
