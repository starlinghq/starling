"""Weight-free checks of declared BERT contracts, complete evidence, and safe orchestration."""

import array
import copy
import json
import sys
import types
from pathlib import Path

import pytest

from model_pipeline import fixtures
from model_pipeline import onboarding_bert as bert
from model_pipeline.onboarding_runner import PipelineError, Stage


def candidate(repo="sentence-transformers/all-MiniLM-L6-v2"):
    spec = bert.CONTRACTS[repo]
    task = spec["task"]
    return {
        "format_version": 1,
        "id": "test-bert-cpu",
        "task": task,
        "origin": {
            "repo": repo,
            "revision": "a" * 40,
            "files": [
                {"path": "onnx/model.onnx", "size": 1, "sha256": "1" * 64},
                {"path": "tokenizer.json", "size": 1, "sha256": "2" * 64},
            ],
        },
        "runtime": {"backend": "onnx", "version": "1.30.0", "provider": "CPUExecutionProvider", "precision": "fp32"},
        "behavior": {
            "architecture": "bert",
            "graph_file": "onnx/model.onnx",
            "tokenizer_file": "tokenizer.json",
            "max_tokens": spec["max_tokens"],
            "padding_side": "right",
            "pad_token_id": spec["pad_token_id"],
            "vocabulary_size": spec["vocabulary_size"],
            "hidden_size": spec["hidden_size"],
            "truncation": "longest_first",
            "tokenizer_profile": spec["tokenizer_profile"],
            "pooling": spec["pooling"],
            "normalize": spec["normalize"],
            "output_name": spec["output_name"],
            "prompts": spec["prompts"],
            **({"score_activation": "identity"} if task == "reranking" else {}),
        },
        "evaluation": {"status": "proposed", "policy": copy.deepcopy(bert.POLICY)},
    }


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def evidence(tmp_path, repo="sentence-transformers/all-MiniLM-L6-v2"):
    contract = candidate(repo)
    profile = tmp_path / "candidate.json"
    write(profile, contract)
    embedding = contract["task"] == "embeddings"
    width = contract["behavior"]["hidden_size"]
    cls_id = bert.CONTRACTS[repo]["cls_token_id"]
    sep_id = bert.CONTRACTS[repo]["sep_token_id"]
    output = [1.0] + [0.0] * (width - 1) if embedding else [3.25]
    rows = [
        {
            "id": "short",
            "text": "",
            "pair": "" if not embedding else None,
            "prompt": "none",
            "input_ids": [cls_id, sep_id],
            "attention_mask": [1, 1],
            "token_type_ids": [0, 0],
            "output": output,
        },
        {
            "id": "longer",
            "text": "hello",
            "pair": "world" if not embedding else None,
            "prompt": "none",
            "input_ids": [cls_id, 200, sep_id],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
            "output": output,
        },
    ]
    if not embedding:
        for row in rows:
            row.update(logits=output, score=output[0], original_score=output[0])
    documents, paths = {}, {}
    for stage in ("reference", "onnx", "dotnet"):
        directory = tmp_path / stage
        directory.mkdir()
        path = directory / f"{stage}.json"
        paths[stage] = path
        doc = {
            "format": bert.FORMAT,
            "format_version": 1,
            "stage": stage,
            "profile": contract["id"],
            "task": contract["task"],
            "candidate_sha256": fixtures.sha256_file(profile),
            "assets": [{"path": entry["path"], "sha256": entry["sha256"]} for entry in contract["origin"]["files"]],
            "environment": {"provider": "CPUExecutionProvider", "onnxruntime": "1.30.0", "machine": "same-host"},
            "cases": copy.deepcopy(rows),
            "batches": [],
        }
        for batch_path in ("ids", "text") if stage == "dotnet" else ("ids",):
            doc["batches"].append(
                {
                    "path": batch_path,
                    "case_ids": ["short", "longer"],
                    "input_lengths": [2, 3],
                    "output": [output, output],
                    "chunks": [{"case_ids": ["short", "longer"], "input_lengths": [2, 3], "width": 3}],
                }
            )
        if stage == "dotnet":
            doc["checks"] = {"pre_cancelled_request": True}
            for row in doc["cases"]:
                row["text_output"] = output
                if not embedding:
                    row["text_score"] = output[0]
        if embedding:
            hidden = array.array("f", output * 5)
            hidden_file = directory / "hidden.f32"
            hidden_file.write_bytes(hidden.tobytes())
            doc["hidden"] = {
                "file": "hidden.f32",
                "sha256": fixtures.sha256_file(hidden_file),
                "cases": [
                    {"id": "short", "offset": 0, "shape": [2, width]},
                    {"id": "longer", "offset": 2 * width, "shape": [3, width]},
                ],
            }
        if stage != "reference":
            doc["reference_sha256"] = fixtures.sha256_file(paths["reference"])
        documents[stage] = doc
        write(path, doc)
    return contract, profile, documents, paths


def evaluate(data):
    contract, profile, _, paths = data
    return bert.evaluate_development(contract, profile, paths["reference"], paths["onnx"], paths["dotnet"])


@pytest.mark.parametrize("repo", list(bert.CONTRACTS))
def test_complete_development_evidence_never_adopts_protected_gates(tmp_path, repo):
    report = evaluate(evidence(tmp_path, repo))
    assert report["status"] == "passed"
    assert report["case_count"] == 2
    assert report["evaluation_status"] == "proposed"
    assert report["protected_release_qualification"] == {"status": "unapproved", "G1": "not-run", "G2": "not-run"}


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "shape", "nonfinite", "identity", "types", "batch", "chunks", "cancellation"]
)
def test_incomplete_or_wrong_evidence_is_rejected(tmp_path, mutation):
    data = evidence(tmp_path)
    _, _, documents, paths = data
    doc = documents["dotnet"]
    if mutation == "missing":
        doc["cases"].pop()
    elif mutation == "duplicate":
        doc["cases"][1]["id"] = "short"
    elif mutation == "shape":
        doc["cases"][0]["output"] = [1.0]
    elif mutation == "nonfinite":
        doc["cases"][0]["text_output"][0] = float("nan")
    elif mutation == "identity":
        doc["assets"][0]["sha256"] = "9" * 64
    elif mutation == "types":
        doc["cases"][1]["token_type_ids"][1] = 1
    elif mutation == "batch":
        doc["batches"].pop()
    elif mutation == "chunks":
        doc["batches"][0].pop("chunks")
    else:
        doc["checks"]["pre_cancelled_request"] = False
    write(paths["dotnet"], doc)
    with pytest.raises(PipelineError):
        evaluate(data)


def test_changed_numeric_results_produce_failed_report(tmp_path):
    data = evidence(tmp_path)
    _, _, documents, paths = data
    documents["dotnet"]["cases"][0]["text_output"] = [0.0, 1.0] + [0.0] * 382
    write(paths["dotnet"], documents["dotnet"])
    report = evaluate(data)
    assert report["status"] == "failed"
    assert "dotnet-text/short/vs_reference" in report["failed_checks"]


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "shape", "extra", "nonfinite"])
def test_hidden_tensor_evidence_is_required_and_verified(tmp_path, mutation):
    data = evidence(tmp_path)
    _, _, documents, paths = data
    doc = documents["dotnet"]
    if mutation == "missing":
        doc.pop("hidden")
    elif mutation == "shape":
        doc["hidden"]["cases"][0]["shape"] = [2, 768]
    elif mutation == "extra":
        doc["hidden"]["cases"].append(doc["hidden"]["cases"][0])
    else:
        file = paths["dotnet"].parent / "hidden.f32"
        file.write_bytes(array.array("f", [float("inf")]).tobytes() if mutation == "nonfinite" else b"bad")
        if mutation == "nonfinite":
            doc["hidden"]["sha256"] = fixtures.sha256_file(file)
    write(paths["dotnet"], doc)
    with pytest.raises(PipelineError):
        evaluate(data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_tokens", 2048),
        ("pooling", "cls"),
        ("padding_side", "left"),
        ("vocabulary_size", 250037),
        ("hidden_size", 768),
    ],
)
def test_incompatible_behavior_cannot_use_fixed_tools(key, value):
    contract = candidate()
    contract["behavior"][key] = value
    with pytest.raises(PipelineError, match="unsupported BERT behavior"):
        bert.validate_candidate_behavior(contract)


def test_nomic_requires_its_hidden_size():
    contract = candidate("nomic-ai/nomic-embed-text-v1.5")
    contract["behavior"]["hidden_size"] = 384
    with pytest.raises(PipelineError, match="unsupported BERT behavior"):
        bert.validate_candidate_behavior(contract)


XLMR_RERANKER = "Starlinghq/bge-reranker-v2-m3-onnx"


def test_xlmr_reranker_contract_is_accepted_and_scores_raw_logits():
    contract = candidate(XLMR_RERANKER)
    bert.validate_candidate_behavior(contract)
    spec = bert.CONTRACTS[XLMR_RERANKER]
    assert (contract["task"], contract["behavior"]["max_tokens"], contract["behavior"]["pooling"]) == (
        "reranking",
        8192,
        "none",
    )
    geometry = (spec["cls_token_id"], spec["sep_token_id"], spec["pad_token_id"], spec["token_type_ids"])
    assert geometry == (0, 2, 1, False)
    assert contract["behavior"]["output_name"] == "logits"
    assert contract["behavior"]["score_activation"] == "identity"
    # The pinned config declares no activation, so the reference passes identity explicitly; ms-marco's config does.
    assert spec["score_activation_source"] == "explicit"
    assert "score_activation_source" not in bert.CONTRACTS["cross-encoder/ms-marco-MiniLM-L6-v2"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("pooling", "cls"),
        ("max_tokens", 512),
        ("hidden_size", 384),
        ("pad_token_id", 0),
        ("tokenizer_profile", "bert-wordpiece-uncased"),
        ("output_name", "token_embeddings"),
        ("score_activation", "sigmoid"),
        ("prompts", {"none": "", "query": bert.QUERY}),
    ],
)
def test_xlmr_reranker_near_misses_are_refused(key, value):
    contract = candidate(XLMR_RERANKER)
    contract["behavior"][key] = value
    with pytest.raises(PipelineError, match="unsupported BERT behavior"):
        bert.validate_candidate_behavior(contract)


def test_xlmr_reranker_refuses_embeddings_task():
    contract = candidate(XLMR_RERANKER)
    contract["task"] = "embeddings"
    with pytest.raises(PipelineError, match="unsupported BERT behavior"):
        bert.validate_candidate_behavior(contract)


def test_policies_cannot_be_raised_after_reference_execution():
    contract = candidate()
    contract["evaluation"]["policy"]["embedding_max_abs"] = 0.1
    with pytest.raises(PipelineError, match="evaluation policy"):
        bert.validate_candidate_behavior(contract)


def test_per_model_fidelity_bounds_may_be_loosened_within_caps():
    contract = candidate("nomic-ai/nomic-embed-text-v1.5")
    contract["evaluation"]["policy"]["embedding_max_abs"] = 5e-5
    contract["evaluation"]["policy"]["hidden_max_abs"] = 1e-3
    bert.validate_candidate_behavior(contract)


def test_fidelity_bounds_cannot_be_loosened_beyond_caps():
    contract = candidate()
    contract["evaluation"]["policy"]["hidden_max_abs"] = 1.0
    with pytest.raises(PipelineError, match="evaluation policy"):
        bert.validate_candidate_behavior(contract)


def test_correctness_bounds_cannot_be_loosened():
    contract = candidate()
    contract["evaluation"]["policy"]["same_backend_max_abs"] = 1e-3
    with pytest.raises(PipelineError, match="evaluation policy"):
        bert.validate_candidate_behavior(contract)


def test_reranker_identity_activation_and_public_reference_are_required(tmp_path):
    data = evidence(tmp_path, "cross-encoder/ms-marco-MiniLM-L6-v2")
    _, _, documents, paths = data
    documents["dotnet"]["cases"][0]["score"] = 0.95
    write(paths["dotnet"], documents["dotnet"])
    with pytest.raises(PipelineError, match="activation"):
        evaluate(data)


def test_distribution_is_explicitly_rejected_until_consumer_integration(tmp_path):
    with pytest.raises(PipelineError, match="distribution stage is not wired"):
        bert.build_stages(
            {}, repo_root=tmp_path, snapshot=tmp_path, reference_python=Path(sys.executable), distribution=True
        )


def test_plugin_constructs_bounded_commands_and_deletes_transient_build(tmp_path, monkeypatch):
    root, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    root.mkdir()
    snapshot.mkdir()
    profile = root / "candidate.json"
    corpus = root / "corpus.jsonl"
    contract = candidate()
    write(profile, contract)
    corpus.write_text("")
    recipe = {"id": "test", "manifest": profile.name, "reference_corpus": corpus.name}
    helper = types.ModuleType("model_pipeline.onboarding_candidate")
    helper.candidate_stages = lambda *args, **kwargs: (
        contract,
        [Stage(name, (), {"candidate": profile}, {}, lambda out, deps: {}) for name in ("assets", "environment")],
        {},
    )
    helper.code_inputs = lambda *args, **kwargs: {"candidate": profile}
    helper.load_candidate = lambda path: json.loads(path.read_text())
    helper.source_reader = lambda stage, *args: stage
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    commands = []

    def command(argv, **kwargs):
        commands.append((argv, kwargs))
        kwargs["log_path"].write_text("okay")
        if argv[1] == "publish":
            published = Path(argv[argv.index("--output") + 1])
            published.mkdir()
            (published / "Starlings.Bert.Fidelity.Tool.dll").write_bytes(b"fixture")
        return {"returncode": 0}

    monkeypatch.setattr(bert, "run_command", command)
    graph = {
        stage.name: stage
        for stage in bert.build_stages(
            recipe, repo_root=root, snapshot=snapshot, reference_python=Path(sys.executable), threads=2, timeout=11
        )
    }
    assert list(graph) == ["assets", "environment", "reference", "onnx", "dotnet", "evaluate", "review"]
    reference = tmp_path / "reference"
    reference.mkdir()
    graph["reference"].execute(reference, {})
    graph["dotnet"].execute(tmp_path / "dotnet", {"reference": reference})
    assert commands[0][0][-2:] == ["--corpus", str(corpus)]
    dotnet = commands[1][0]
    assert dotnet[1] == "publish" and "-m:1" in dotnet
    assert "-p:RestoreLockedMode=true" in dotnet
    assert not Path(dotnet[dotnet.index("--artifacts-path") + 1]).exists()
    assert commands[2][0][1].endswith("Starlings.Bert.Fidelity.Tool.dll")
    assert all(kwargs["timeout"] == 11 and kwargs["env"]["OMP_NUM_THREADS"] == "2" for _, kwargs in commands)


def _contract_table() -> dict:
    return json.loads(bert.CONTRACTS_PATH.read_text(encoding="utf-8"))


def _first(task: str) -> str:
    """The first repository of the table with the task (the tests name no model)."""
    return next(repo for repo, spec in bert.CONTRACTS.items() if spec["task"] == task)


def test_contract_table_is_the_file_the_dotnet_package_embeds():
    root = Path(__file__).resolve().parents[2]
    project = (root / "ai-dotnet/src/Starlings.Models.Bert/Starlings.Models.Bert.csproj").read_text(encoding="utf-8")
    assert 'Include="../../../model-pipeline/routes/bert-onnx-cpu-v1/contracts.json"' in project
    assert root / "model-pipeline/routes/bert-onnx-cpu-v1/contracts.json" == bert.CONTRACTS_PATH
    assert list(bert.CONTRACTS) == list(_contract_table()["contracts"])
    queries = [spec["prompts"]["query"] for spec in bert.CONTRACTS.values() if "query" in spec["prompts"]]
    assert queries and set(queries) == {bert.QUERY}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("task", "classification"),
        ("pooling", "max"),
        ("normalize", "l1"),
        ("tokenizer_profile", "gpt2-bpe"),
        ("output_name", ""),
        ("max_tokens", 0),
        ("pad_token_id", 30522),
        ("cls_token_id", -1),
        ("hidden_size", True),
        ("token_type_ids", 1),
        ("score_activation_source", "config"),
        ("unknown_field", 1),
    ],
)
def test_contract_table_rejects_an_embeddings_rule_break(key, value):
    table = _contract_table()
    table["contracts"][_first("embeddings")][key] = value
    with pytest.raises(PipelineError):
        bert.validate_contracts(table)


@pytest.mark.parametrize(("key", "value"), [("pooling", "cls"), ("normalize", "l2"), ("score_activation_source", "x")])
def test_contract_table_rejects_a_reranking_rule_break(key, value):
    table = _contract_table()
    table["contracts"][_first("reranking")][key] = value
    with pytest.raises(PipelineError):
        bert.validate_contracts(table)


def test_contract_table_rejects_missing_fields_prompts_keys_format_and_duplicates(tmp_path):
    repo = next(iter(bert.CONTRACTS))
    missing = _contract_table()
    del missing["contracts"][repo]["output_name"]
    prompts = _contract_table()
    prompts["contracts"][repo]["prompts"] = {"query": "q: "}
    bad_key = _contract_table()
    bad_key["contracts"]["not-a-repository"] = bad_key["contracts"][repo]
    version = _contract_table()
    version["format_version"] = 2
    for table in (missing, prompts, bad_key, version):
        with pytest.raises(PipelineError):
            bert.validate_contracts(table)
    text = bert.CONTRACTS_PATH.read_text(encoding="utf-8")
    entry = json.dumps(_contract_table()["contracts"][repo])
    duplicated = tmp_path / "contracts.json"
    key = json.dumps(repo) + ": "
    duplicated.write_text(text.replace(key + "{", f"{key}{entry}, {key}{{", 1), "utf-8")
    with pytest.raises(PipelineError, match="duplicate"):
        bert.load_contracts(duplicated)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(PipelineError):
        bert.load_contracts(malformed)
