"""Weight-free checks of model-pipeline scaffold on a copy of the files it edits."""

import json
import shutil
from pathlib import Path

import pytest

from model_pipeline import onboarding_bert as bert
from model_pipeline import scaffold
from model_pipeline.cli import main
from model_pipeline.onboarding_runner import PipelineError

REPO = Path(__file__).resolve().parents[2]
CANDIDATES = REPO / "model-pipeline" / "candidates"


def _profile(name: str) -> dict:
    return json.loads((CANDIDATES / name / "profile.json").read_text(encoding="utf-8"))


# Every committed candidate of the BERT route, found by its contract rather than by name.
BERT_CANDIDATES = sorted(
    path.parent.name
    for path in CANDIDATES.glob("*/profile.json")
    if _profile(path.parent.name).get("behavior", {}).get("architecture") == "bert"
)
# A WordPiece embeddings candidate to copy as the new one.
TEMPLATE = next(
    name
    for name in BERT_CANDIDATES
    if _profile(name)["task"] == "embeddings"
    and _profile(name)["behavior"]["tokenizer_profile"] == "bert-wordpiece-uncased"
)
NEW = "new-encoder"
NEW_REPO = "example/new-encoder"


def test_the_route_has_committed_candidates():
    assert len(BERT_CANDIDATES) == len(bert.CONTRACTS)


@pytest.mark.parametrize("name", BERT_CANDIDATES)
def test_every_committed_bert_candidate_is_already_fully_scaffolded(name):
    result = scaffold.scaffold(REPO, name)
    assert [piece["action"] for piece in result["files"]] == ["unchanged"] * 5 + ["print"]
    assert result["repository"] in bert.CONTRACTS


def _copy_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in (
        scaffold.CONTRACTS,
        scaffold.FETCH_SCRIPT,
        scaffold.README,
        scaffold.CONTRACT_TESTS,
        scaffold.CORPORA["embeddings"],
        scaffold.CORPORA["reranking"],
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, root / relative)
    (root / "model-pipeline/protected").mkdir(parents=True)
    return root


def _new_candidate(root: Path, repo: str = NEW_REPO, **behavior) -> dict:
    profile = _profile(TEMPLATE)
    profile["id"] = f"{NEW}/onnx-fp32-cpu"
    profile["origin"]["repo"] = repo
    profile["behavior"].update({"max_tokens": 128, **behavior})
    target = root / f"model-pipeline/candidates/{NEW}/profile.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def test_scaffold_writes_every_piece_once(tmp_path):
    root = _copy_repo(tmp_path)
    profile = _new_candidate(root)
    behavior = profile["behavior"]
    dry = scaffold.scaffold(root, NEW)
    assert [piece["action"] for piece in dry["files"]] == ["create", "insert", "insert", "insert", "insert", "print"]
    assert not (root / f"model-pipeline/recipes/{NEW}-cpu.yaml").exists()

    written = scaffold.scaffold(root, NEW, write=True)
    actions = [piece["action"] for piece in written["files"]][:5]
    assert actions == ["created", "inserted", "inserted", "inserted", "inserted"]
    recipe = (root / f"model-pipeline/recipes/{NEW}-cpu.yaml").read_text(encoding="utf-8")
    assert "plugin: bert-onnx-cpu-v1\n" in recipe
    assert f"manifest: model-pipeline/candidates/{NEW}/profile.json" in recipe
    contracts = bert.load_contracts(root / scaffold.CONTRACTS)
    assert list(contracts)[-1] == NEW_REPO
    assert contracts[NEW_REPO]["max_tokens"] == 128
    assert (contracts[NEW_REPO]["cls_token_id"], contracts[NEW_REPO]["sep_token_id"]) == (101, 102)
    fetch = (root / scaffold.FETCH_SCRIPT).read_text(encoding="utf-8").split("\n")
    line = f'    "{NEW_REPO}|{profile["origin"]["revision"]}|tokenizer.json|'
    index = next(i for i, text in enumerate(fetch) if text.startswith(line))
    assert fetch[index + 1] == ")"
    readme = (root / scaffold.README).read_text(encoding="utf-8")
    row = (
        f"| `{NEW}` | `{NEW}/onnx-fp32-cpu` | embeddings | 128 | {behavior['pooling']} | {behavior['hidden_size']} | "
        f"{behavior['normalize']} | "
    )
    assert row in readme
    tests = (root / scaffold.CONTRACT_TESTS).read_text(encoding="utf-8").split("\n")
    anchor = next(i for i, text in enumerate(tests) if text.startswith(scaffold._TEST_ANCHOR))
    expected = (
        f'    [InlineData("{NEW_REPO}", 128, "{behavior["pooling"]}", BertTask.Embeddings, {behavior["hidden_size"]})]'
    )
    assert tests[anchor - 1] == expected

    again = scaffold.scaffold(root, NEW, write=True)
    assert [piece["action"] for piece in again["files"]][:5] == ["unchanged"] * 5


def test_scaffold_refuses_a_candidate_that_contradicts_the_table(tmp_path):
    root = _copy_repo(tmp_path)
    _new_candidate(root, repo=_profile(TEMPLATE)["origin"]["repo"])
    with pytest.raises(PipelineError, match="different"):
        scaffold.scaffold(root, NEW)


def test_scaffold_refuses_other_routes_and_paths(tmp_path):
    root = _copy_repo(tmp_path)
    _new_candidate(root, architecture="qwen3")
    with pytest.raises(PipelineError):
        scaffold.scaffold(root, NEW)
    with pytest.raises(PipelineError):
        scaffold.scaffold(root, f"../{TEMPLATE}")
    with pytest.raises(PipelineError):
        scaffold.scaffold(root, "missing")
    outside = tmp_path / "elsewhere" / "profile.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PipelineError):
        scaffold.scaffold(root, str(outside))


def test_scaffold_refuses_a_contract_the_route_does_not_implement(tmp_path):
    root = _copy_repo(tmp_path)
    _new_candidate(root, pooling="max")
    with pytest.raises(PipelineError):
        scaffold.scaffold(root, NEW, write=True)
    table = json.loads((root / scaffold.CONTRACTS).read_text(encoding="utf-8"))
    assert NEW_REPO not in table["contracts"]


def test_scaffold_cli_prints_json(tmp_path, capsys):
    root = _copy_repo(tmp_path)
    _new_candidate(root)
    assert main(["scaffold", NEW, "--repo-root", str(root)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["written"] is False and report["plugin"] == "bert-onnx-cpu-v1"
    assert main(["scaffold", "missing", "--repo-root", str(root)]) == 2
