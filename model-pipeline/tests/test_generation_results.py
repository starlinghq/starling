"""Stage results format version 3 (task generation): schema, converters and the evaluator on v3 files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_evaluate_generation import PROFILE, VOCAB, Repo, status

from model_pipeline import evaluate, generation_results, results
from model_pipeline.cli import main
from model_pipeline.results import ResultsError, ToleranceOverrideError

FIXTURE_NAME = "generation.jsonl"


def convert(repo: Repo, tmp_path: Path, name: str = "v3", **kwargs) -> Path:
    out = tmp_path / name
    generation_results.convert_runner_directory(repo.results, out, PROFILE, **kwargs)
    return out


def run_v3(repo: Repo, directory: Path, tensors: bool = True) -> evaluate.Evaluation:
    return evaluate.evaluate(
        repo.manifest,
        directory,
        repo_root=repo.root,
        protected_base="HEAD",
        reference_tensors=repo.tensors if tensors else None,
    )


def checks_of(ev) -> dict[str, tuple]:
    return {c["id"]: (c["status"], c["metrics"], c["detail"]) for c in ev.report["checks"]}


def test_v3_equals_runner_directory(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(edit=lambda r: r.update(output_ids=[4, 6, 7]) if r["id"] == "a" else None)
    direct = repo.run()
    converted = run_v3(repo, convert(repo, tmp_path))
    assert checks_of(converted) == checks_of(direct)
    assert converted.report["gates"] == direct.report["gates"]
    assert converted.report["gates"]["G2"]["status"] == "passed"
    assert converted.report["greedy_divergences"] == [{"id": "a", "step": 1, "reference_margin": 0.0005}]
    assert converted.exit_code == direct.exit_code == 0
    names = set(converted.report["inputs"]["results"])
    assert {"template.results.json", "dotnet.results.json", "step0.f32"} <= names


def test_v3_failures_are_detected(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(edit=lambda r: r.update(rendered="<u>HI</u>", output_ids=[6, 5]) if r["id"] == "b" else None)
    ev = run_v3(repo, convert(repo, tmp_path))
    assert status(ev, "tokenizer.template_rendering")["status"] == "failed"
    assert status(ev, "dotnet-ids.greedy_ids")["status"] == "failed"
    assert ev.exit_code == evaluate.EXIT_FAILED


def test_v3_files_follow_the_schema(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = convert(repo, tmp_path)
    template = json.loads((out / "template.results.json").read_text(encoding="utf-8"))
    dotnet = json.loads((out / "dotnet.results.json").read_text(encoding="utf-8"))
    assert (template["format_version"], template["task"], template["stage"]) == (3, "generation", "template")
    assert set(dotnet["sections"]) == {"cached_vs_uncached", "chunked_prefill", "concurrency"}
    assert dotnet["cases"]["a"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "consistent": True}
    assert dotnet["tensors"]["step0_logprobs"]["cases"][0] == {"id": "a", "shape": [VOCAB]}
    assert "dotnet" in {results.load(out / "dotnet.results.json").stage}


def valid_document(**changes) -> dict:
    doc = results.new_results(
        profile=PROFILE, stage="dotnet", candidate="c", producer={"name": "p"}, environment={}, assets=[],
        task="generation",
    )  # fmt: skip
    doc.update(changes)
    return doc


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "tokenizer"},
        {"task": "embedding"},
        {"batches": {"x": {"output": "o", "case_ids": [], "values": []}}},
        {"format_version": 2},
        {"format_version": 1},
    ],
)
def test_schema_rejects(tmp_path, changes):
    doc = valid_document(**changes)
    if changes.get("format_version") == 1:
        doc.pop("task")
        doc["sections"] = {}
    path = tmp_path / "x.results.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ResultsError):
        results.load(path)


def test_schema_still_accepts_old_versions(tmp_path):
    for doc in (
        results.new_results(
            profile="a/b", stage="dotnet", candidate="c", producer={"name": "p"}, environment={}, assets=[]
        ),
        results.new_results(
            profile="a/b", stage="tokenizer", candidate="c", producer={"name": "p"}, environment={}, assets=[],
            task="entity-extraction",
        ),
    ):  # fmt: skip
        path = tmp_path / "x.results.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert results.load(path).format_version in (1, 2)
    assert results.load(_write(tmp_path, valid_document())).format_version == 3


def _write(directory: Path, doc: dict) -> Path:
    path = directory / "y.results.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_v3_without_sidecar_checks_best_pairs_and_leaves_g2_open(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = convert(repo, tmp_path)
    doc = json.loads((out / "dotnet.results.json").read_text(encoding="utf-8"))
    doc["tensors"] = {}
    (out / "dotnet.results.json").write_text(json.dumps(doc), encoding="utf-8")
    (out / "step0.f32").unlink()
    ev = run_v3(repo, out)
    assert status(ev, "dotnet-ids.step0_argmax")["status"] == "passed"
    assert status(ev, "dotnet-ids.step0_top5_logprob")["status"] == "passed"
    assert status(ev, "dotnet-ids.step0_logprob")["status"] == "not-run"
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_v3_best_pairs_catch_a_wrong_argmax(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = convert(repo, tmp_path)
    doc = json.loads((out / "dotnet.results.json").read_text(encoding="utf-8"))
    doc["tensors"] = {}
    doc["cases"]["a"]["step0_top"] = [[0, -0.1]] + [[i, -5.0] for i in range(1, 5)]
    (out / "dotnet.results.json").write_text(json.dumps(doc), encoding="utf-8")
    ev = run_v3(repo, out)
    assert status(ev, "dotnet-ids.step0_argmax")["status"] == "failed"


def test_v3_sidecar_digest_is_checked(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = convert(repo, tmp_path)
    data = bytearray((out / "step0.f32").read_bytes())
    data[0] ^= 1
    (out / "step0.f32").write_bytes(bytes(data))
    with pytest.raises(evaluate.EvaluationError, match="SHA-256"):
        run_v3(repo, out)


def test_v3_needs_both_stages(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = convert(repo, tmp_path)
    (out / "template.results.json").unlink()
    with pytest.raises(evaluate.EvaluationError, match="stage file"):
        run_v3(repo, out)


def test_v3_may_not_carry_thresholds(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = convert(repo, tmp_path)
    doc = json.loads((out / "dotnet.results.json").read_text(encoding="utf-8"))
    doc["sections"]["cached_vs_uncached"]["tolerance"] = 1.0
    (out / "dotnet.results.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ToleranceOverrideError):
        run_v3(repo, out)
    code = main(
        [
            "evaluate",
            str(repo.manifest),
            "--results",
            str(out),
            "--out",
            str(tmp_path / "o"),
            "--repo-root",
            str(repo.root),
        ]
    )
    assert code == evaluate.EXIT_PROTECTED


def test_v3_profile_must_match_the_manifest(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = tmp_path / "other"
    generation_results.convert_runner_directory(repo.results, out, "synthetic-llm/other")
    ev = run_v3(repo, out)
    assert status(ev, "dotnet-ids.identity")["status"] == "invalid"


def test_semantics_of_another_fixture_are_refused_in_v3(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(semantics={"cached_vs_uncached": {"prompt": "en-list", "max_abs_logprob": 3e-5}})
    with pytest.raises(evaluate.EvaluationError, match="not in the fixture"):
        run_v3(repo, convert(repo, tmp_path))


def test_fixture_converter_gives_the_reference_side(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = tmp_path / "ref"
    generation_results.convert_fixture(repo.protected / "reference" / FIXTURE_NAME, out, PROFILE)
    ev = run_v3(repo, out)
    for check in (
        "tokenizer.template_rendering",
        "tokenizer.prompt_ids",
        "dotnet-ids.greedy_ids",
        "dotnet-ids.finish_reason",
        "dotnet-ids.step0_argmax",
        "dotnet-ids.step0_top5_logprob",
    ):
        assert status(ev, check)["status"] == "passed", check
    # What only a runner observes is not invented from the fixture.
    assert status(ev, "dotnet-ids.streaming_reassembly")["status"] == "failed"
    assert ev.report["gates"]["G2"]["status"] == "failed"
    assert "t" in ev.report["excluded_by_contract"]


def test_cli_convert_generation(tmp_path, capsys):
    repo = Repo(tmp_path)
    repo.write_run()
    out = tmp_path / "cli"
    assert main(["convert-generation", "--runner", str(repo.results), "--profile", PROFILE, "--out", str(out)]) == 0
    assert (out / "dotnet.results.json").is_file() and (out / "step0.f32").is_file()
    fx = repo.protected / "reference" / FIXTURE_NAME
    assert main(["convert-generation", "--fixture", str(fx), "--profile", PROFILE, "--out", str(tmp_path / "f")]) == 0
    assert (
        main(["convert-generation", "--runner", str(tmp_path / "nope"), "--profile", PROFILE, "--out", str(out)]) == 2
    )
    capsys.readouterr()
