"""Weight-free checks of the Kev decision route: contract allow-list, registration, evaluator and orchestration."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from model_pipeline import evaluate_decision, fixtures, onboarding_candidate, onboarding_plugins
from model_pipeline import onboarding_kev as kev
from model_pipeline.cli import main
from model_pipeline.onboarding_candidate import load_candidate
from model_pipeline.onboarding_runner import PipelineError

REPO = Path(__file__).resolve().parents[2]
PROFILES = {
    "jaredpalmer/kev-0.8b": REPO / "model-pipeline/candidates/kev-0.8b/profile.json",
    "jaredpalmer/kev-4b": REPO / "model-pipeline/candidates/kev-4b/profile.json",
}
TOLERANCES = REPO / "model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml"


def profile(repo: str = "jaredpalmer/kev-0.8b") -> dict:
    return json.loads(PROFILES[repo].read_text(encoding="utf-8"))


# --- contract -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("repo", sorted(PROFILES))
def test_shipped_candidates_pass_schema_and_exact_contract(repo):
    candidate = load_candidate(PROFILES[repo])
    contract = kev.validate_candidate_behavior(candidate, REPO)
    assert candidate["id"] == contract["id"]
    assert candidate["behavior"]["delimiters"] == {
        "state": 248060,
        "question": 248061,
        "option": 248049,
        "option_end": 248050,
        "decide": 248062,
    }
    assert candidate["behavior"]["pad_token_id"] == 248044
    assert set(candidate["behavior"]["capabilities"].values()) <= {
        "supported",
        "unsupported",
        "experimental",
        "untested",
    }


def test_contract_values_per_repository():
    small, large = (kev.CONTRACTS[r]["behavior"] for r in ("jaredpalmer/kev-0.8b", "jaredpalmer/kev-4b"))
    assert (small["hidden_size"], small["num_hidden_layers"], small["num_key_value_heads"]) == (1024, 24, 2)
    assert (large["hidden_size"], large["num_hidden_layers"], large["num_key_value_heads"]) == (2560, 32, 4)
    assert (small["linear_num_value_heads"], large["linear_num_value_heads"]) == (16, 32)
    assert small["temperature"] == 2.3510958125672174
    assert large["temperature"] == 2.406050072164233
    for behavior in (small, large):
        assert behavior["head_dim"] == behavior["pointer_dim"] == 256
        assert behavior["max_state_tokens"] == behavior["max_row_tokens"] == 8192


def test_policy_equals_binding_kev_tolerances():
    bound = evaluate_decision.protected_policy(TOLERANCES.read_text(encoding="utf-8"))
    assert bound == {key: evaluate_decision.POLICY[key] for key in bound}
    assert bound["probability_max_abs"] == 2.0e-4 and bound["argmax_flip_margin_exception"] == 1.0e-4


def test_protected_policy_refuses_a_missing_margin_exception():
    text = TOLERANCES.read_text(encoding="utf-8").replace("top-two margin is below", "margin under")
    with pytest.raises(PipelineError, match="margin exception"):
        evaluate_decision.protected_policy(text)


NEAR_MISSES = [
    ("hidden size", lambda c: c["behavior"].update(hidden_size=2048), "behavior.hidden_size"),
    ("layers", lambda c: c["behavior"].update(num_hidden_layers=28), "behavior.num_hidden_layers"),
    ("temperature", lambda c: c["behavior"].update(temperature=2.351095812567217), "behavior.temperature"),
    ("integer temperature", lambda c: c["behavior"].update(temperature=2), "behavior.temperature"),
    ("delimiter", lambda c: c["behavior"]["delimiters"].update(decide=248063), "behavior.delimiters"),
    ("pad", lambda c: c["behavior"].update(pad_token_id=0), "behavior.pad_token_id"),
    ("state limit", lambda c: c["behavior"].update(max_state_tokens=65536), "behavior.max_state_tokens"),
    ("capability", lambda c: c["behavior"]["capabilities"].update(bf16="experimental"), "behavior.capabilities"),
    ("extra behavior", lambda c: c["behavior"].update(pooling="mean"), "behavior"),
    ("tag revision", lambda c: c["origin"].update(revision="v1.0"), "origin.revision"),
    (
        "other revision",
        lambda c: c["origin"].update(revision="bf75a6a8848ea6960ff2ed108d9ed44c2941174f"),
        "origin.revision",
    ),
    ("base branch", lambda c: c["origin"]["base"].update(revision="main"), "origin.base.revision"),
    ("code tag", lambda c: c["origin"]["code"].update(revision="kev-1.0"), "origin.code"),
    ("provider", lambda c: c["runtime"].update(provider="CUDAExecutionProvider"), "runtime"),
    ("precision", lambda c: c["runtime"].update(precision="bf16"), "runtime"),
    ("repository", lambda c: c["origin"].update(repo="jaredpalmer/kev-9b"), "origin.repo"),
    ("identifier", lambda c: c.update(id="kev-0.8b/onnx-fp32-cuda"), "id"),
    ("task", lambda c: c.update(task="embeddings"), "task"),
    ("missing file", lambda c: c["origin"]["files"].pop(), "origin.files"),
    ("base tokenizer pinned as Kev's", lambda c: c["origin"]["files"][-2].update(sha256="f" * 64), "origin.files"),
    ("loosened bound", lambda c: c["evaluation"]["policy"].update(probability_max_abs=1e-3), "evaluation.policy"),
    ("dropped flip rule", lambda c: c["evaluation"]["policy"].pop("argmax_flips"), "evaluation.policy"),
]


@pytest.mark.parametrize(("name", "mutate", "field"), NEAR_MISSES, ids=[case[0] for case in NEAR_MISSES])
def test_contract_refuses_near_misses_field_by_field(name, mutate, field):
    candidate = profile()
    assert candidate["origin"]["files"][-2]["path"] == "tokenizer.json"
    mutate(candidate)
    with pytest.raises(PipelineError, match=field.replace(".", r"\.")):
        kev.validate_candidate_behavior(candidate)


def test_contract_does_not_mix_repositories():
    candidate = profile("jaredpalmer/kev-4b")
    candidate["behavior"] = profile()["behavior"]
    with pytest.raises(PipelineError, match="behavior.hidden_size"):
        kev.validate_candidate_behavior(candidate)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["origin"].update(revision="main"),
        lambda c: c["origin"]["base"].update(revision="v1.0"),
        lambda c: c["origin"]["code"].update(revision="kev-1.0"),
        lambda c: c["runtime"].update(provider="CUDAExecutionProvider"),
        lambda c: c["origin"]["code"].update(repository="git@github.com:jaredpalmer/kev"),
        lambda c: c["origin"]["base"].update(extra=True),
    ],
)
def test_schema_refuses_branches_tags_and_unknown_fields(tmp_path, mutate):
    candidate = profile()
    mutate(candidate)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(PipelineError, match="invalid candidate contract"):
        load_candidate(path)


def test_protected_bounds_are_cross_checked(tmp_path):
    root = tmp_path
    path = root / kev.CONTRACTS["jaredpalmer/kev-0.8b"]["protected_tolerances"]
    path.parent.mkdir(parents=True)
    path.write_text(TOLERANCES.read_text(encoding="utf-8").replace("2.0e-4", "1.0e-4"), encoding="utf-8")
    with pytest.raises(PipelineError, match="binding"):
        kev.validate_candidate_behavior(profile(), root)


# --- registration ---------------------------------------------------------------------------------------------------


def test_plugin_is_registered_and_recipes_plan(capsys):
    assert onboarding_plugins.get_plugin({"plugin": "kev-onnx-cpu-v1"}) is kev
    for name, identifier in (
        ("kev-0.8b-cpu.yaml", "kev-0.8b/onnx-fp32-cpu"),
        ("kev-4b-cpu.yaml", "kev-4b/onnx-fp32-cpu"),
    ):
        assert main(["plan", str(REPO / "model-pipeline/recipes" / name), "--repo-root", str(REPO)]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["plugin"] == "kev-onnx-cpu-v1" and plan["profile"] == identifier
        assert plan["stages"] == ["assets", "environment", "reference", "build", "onnx", "dotnet", "evaluate", "review"]
        assert plan["qualification_status"] == "unapproved"
    assert plan["policy_source"].startswith("proposed")


def test_other_plugins_refuse_a_base_snapshot(tmp_path):
    with pytest.raises(PipelineError, match="no base snapshot"):
        onboarding_plugins.build_stages(
            {"plugin": "bert-onnx-cpu-v1"}, repo_root=tmp_path, snapshot=tmp_path, base_snapshot=tmp_path
        )


def test_kev_requires_a_base_snapshot_and_refuses_distribution(tmp_path):
    recipe = {"recipe_version": 1, "plugin": kev.PLUGIN, "manifest": "x"}
    with pytest.raises(PipelineError, match="base-snapshot"):
        kev.build_stages(recipe, repo_root=REPO, snapshot=tmp_path, reference_python=tmp_path)
    with pytest.raises(PipelineError, match="distribution"):
        kev.build_stages(recipe, repo_root=REPO, snapshot=tmp_path, reference_python=tmp_path, distribution=True)


# --- base snapshot in Hugging Face cache layout ----------------------------------------------------------------------


def hf_base(tmp_path: Path, candidate: dict, content: bytes = b"x") -> Path:
    base = candidate["origin"]["base"]
    repository = tmp_path / "hub" / ("models--" + base["repo"].replace("/", "--"))
    snapshot = repository / "snapshots" / base["revision"]
    (repository / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, record in enumerate(base["files"]):
        blob = repository / "blobs" / f"blob{index}"
        blob.write_bytes(content)
        (snapshot / record["path"]).symlink_to(Path("../../blobs") / blob.name)
    return snapshot


def test_base_cache_layout_and_pins(tmp_path):
    candidate = profile()
    import hashlib

    for record in candidate["origin"]["base"]["files"]:
        record.update(size=1, sha256=hashlib.sha256(b"x").hexdigest())
    snapshot = hf_base(tmp_path, candidate)
    base = candidate["origin"]["base"]
    assert kev._base_cache(snapshot, base["repo"], base["revision"]) == tmp_path / "hub"
    assert len(kev.verify_base(candidate, snapshot)) == len(base["files"])
    with pytest.raises(PipelineError, match="snapshots"):
        kev._base_cache(snapshot, base["repo"], "0" * 40)
    with pytest.raises(PipelineError, match="snapshots"):
        kev._base_cache(snapshot, "Qwen/Qwen3.5-4B-Base", base["revision"])
    (snapshot / "config.json").unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"x")
    (snapshot / "config.json").symlink_to(outside)
    with pytest.raises(PipelineError, match="escapes"):
        kev.verify_base(candidate, snapshot)


def test_base_pin_mismatch_is_refused(tmp_path):
    candidate = profile()
    snapshot = hf_base(tmp_path, candidate)
    with pytest.raises(PipelineError, match="base asset mismatch"):
        kev.verify_base(candidate, snapshot)


def test_base_config_and_tokenizer_contract_checks(tmp_path):
    candidate = profile()
    behavior = candidate["behavior"]
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(24)]
    text = {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "full_attention_interval": 4,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "head_dim": 256,
        "layer_types": layer_types,
    }
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": text}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert kev.check_base_config(candidate, tmp_path)["num_hidden_layers"] == 24
    text["layer_types"] = list(reversed(layer_types))
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(PipelineError, match="layer_types"):
        kev.check_base_config(candidate, tmp_path)
    tokens = [{"content": token, "id": behavior["delimiters"][name]} for name, token in kev.DELIMITER_TOKENS.items()]
    tokens.append({"content": "<|endoftext|>", "id": 248044})
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": tokens}), encoding="utf-8")
    assert kev.check_tokenizer(candidate, tmp_path)["pad_token_id"] == 248044
    tokens[0]["id"] = 1
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": tokens}), encoding="utf-8")
    with pytest.raises(PipelineError, match="delimiter"):
        kev.check_tokenizer(candidate, tmp_path)


# --- evaluator with fake stage outputs -------------------------------------------------------------------------------


def softmax(logits):
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    return [value / sum(weights) for value in weights]


def fake_evidence(tmp_path: Path, candidate: dict | None = None) -> dict:
    candidate = candidate or profile()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    digest = fixtures.sha256_file(path)
    questions = {
        "two": [
            {"id": "a", "instr": "first?", "options": ["no", "yes"], "label": 1},
            {"id": "b", "instr": "second?", "options": ["x", "y", "z"], "label": None},
        ],
        "one": [{"id": "only", "instr": "", "options": ["single"], "label": None}],
        "three": [
            {"id": "p", "instr": "p", "options": ["1", "2", "3", "4"], "label": 0},
            {"id": "q", "instr": "q", "options": ["no", "yes"], "label": None},
            {"id": "r", "instr": "near tie", "options": ["left", "right"], "label": None},
        ],
    }
    probabilities = {
        "two": [softmax([0.1, 1.2]), softmax([0.3, -0.2, 2.0])],
        "one": [[1.0]],
        "three": [softmax([2.0, 0.5, 0.1, -1.0]), softmax([0.0, 0.7]), [0.50003, 0.49997]],
    }
    cases, onnx_cases, dotnet_cases = [], [], []
    for identifier, qs in questions.items():
        state = [248060, 100 + len(identifier)]
        rows, start = [], len(state)
        for index, question in enumerate(qs):
            length = 3 + 2 * len(question["options"]) + index
            ids = [248061] + [7] * (length - 2) + [248062]
            rows.append(
                {
                    "ids": ids,
                    "pos": list(range(start, start + length)),
                    "decide": length - 1,
                    "opts": [2 + 2 * k for k in range(len(question["options"]))],
                }
            )
        probs = probabilities[identifier]
        cases.append(
            {
                "id": identifier,
                "record": {"state": identifier, "questions": qs},
                "state_ids": state,
                "rows": rows,
                "probs": probs,
            }
        )
        shifted = [[value + 1e-7 * (k % 2 * 2 - 1) for k, value in enumerate(p)] for p in probs]
        lengths = [len(state) + len(row["ids"]) for row in rows]
        onnx_cases.append(
            {
                "id": identifier,
                "output": shifted,
                "uncached_output": copy.deepcopy(shifted),
                "batch": {"row_lengths": lengths, "width": max(lengths), "output": copy.deepcopy(shifted)}
                if len(rows) > 1
                else None,
            }
        )
        dotnet_cases.append(
            {
                "id": identifier,
                "state_ids": list(state),
                "rows": copy.deepcopy(rows),
                "output": copy.deepcopy(shifted),
                "uncached_output": copy.deepcopy(shifted),
                "text_output": copy.deepcopy(shifted),
            }
        )
    reference = {"profile": candidate["id"], "candidate_sha256": digest, "cases": cases}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    bundle = {
        name: {"size": 1, "sha256": f"{index}" * 64}
        for index, name in enumerate(
            ("model.onnx", "model.onnx.data", "head.safetensors", "head.json", "tokenizer.json")
        )
    }
    models = {name: entry["sha256"] for name, entry in bundle.items()}
    documents = {
        "build": {"profile": candidate["id"], "candidate_sha256": digest, "bundle": bundle, "graph": {"ops": {}}},
        "onnx": {
            "profile": candidate["id"],
            "candidate_sha256": digest,
            "reference_sha256": fixtures.sha256_file(reference_path),
            "environment": {"machine": "host", "onnxruntime": "1.30.0", "model_sha256": models},
            "cases": onnx_cases,
        },
        "dotnet": {
            "profile": candidate["id"],
            "candidate_sha256": digest,
            "reference_sha256": fixtures.sha256_file(reference_path),
            "environment": {
                "machine": "host",
                "onnxruntime": "1.30.0",
                "dotnet": "10",
                "threads": 2,
                "model_sha256": {k: models[k] for k in ("model.onnx", "model.onnx.data")},
            },
            "cases": dotnet_cases,
            "cancellation_verified": True,
        },
    }
    return {"candidate": candidate, "profile": path, "reference": reference_path, "documents": documents}


def run_evaluation(tmp_path: Path, evidence: dict) -> dict:
    paths = {}
    for name, document in evidence["documents"].items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(document), encoding="utf-8")
    return evaluate_decision.evaluate(
        evidence["candidate"],
        evidence["profile"],
        evidence["reference"],
        paths["build"],
        paths["onnx"],
        paths["dotnet"],
    )


def test_evaluation_passes_complete_evidence(tmp_path):
    report = run_evaluation(tmp_path, fake_evidence(tmp_path))
    assert report["status"] == "passed", report["failures"]
    assert report["question_count"] == 6 and report["padded_batches"] == 2
    assert report["accuracy"]["status"] == "identical" and report["accuracy"]["labelled_questions"] == 2
    metrics = {check["metric"] for check in report["checks"]}
    assert metrics == {
        "probability_max_abs",
        "cached_vs_uncached_probability_max_abs",
        "padded_batch_vs_single_probability_max_abs",
        "csharp_vs_python_probability_max_abs",
    }
    assert all(check["max_abs"] <= check["bound"] for check in report["checks"])
    assert all("p99_abs" in check and "mean_abs" in check for check in report["checks"])
    markdown = evaluate_decision.render_markdown(report)
    assert "Status: **passed**" in markdown and "unapproved" in markdown


def test_probability_tolerance_failure(tmp_path):
    evidence = fake_evidence(tmp_path)
    case = evidence["documents"]["dotnet"]["cases"][0]
    case["uncached_output"][1] = [
        value + (3e-4 if k == 0 else -3e-4 if k == 2 else 0) for k, value in enumerate(case["uncached_output"][1])
    ]
    report = run_evaluation(tmp_path, evidence)
    assert report["status"] == "failed"
    failed = {check["id"] for check in report["checks"] if check["status"] == "failed"}
    assert "dotnet-ids-uncached/vs_reference" in failed
    assert "dotnet/cached_vs_uncached" in failed
    assert "dotnet-ids-uncached/vs_python-onnx-uncached" in failed


def test_cached_versus_uncached_bound_is_tighter(tmp_path):
    evidence = fake_evidence(tmp_path)
    case = evidence["documents"]["onnx"]["cases"][2]
    case["output"][0] = [
        value + (2e-5 if k == 0 else -2e-5 if k == 1 else 0) for k, value in enumerate(case["output"][0])
    ]
    report = run_evaluation(tmp_path, evidence)
    failed = {check["id"] for check in report["checks"] if check["status"] == "failed"}
    assert "python-onnx/cached_vs_uncached" in failed
    assert "python-onnx-cached/vs_reference" not in failed


def test_flip_outside_margin_fails(tmp_path):
    evidence = fake_evidence(tmp_path)
    for stage, key in (
        ("onnx", "output"),
        ("onnx", "uncached_output"),
        ("dotnet", "output"),
        ("dotnet", "uncached_output"),
        ("dotnet", "text_output"),
    ):
        evidence["documents"][stage]["cases"][2][key][1] = [0.6, 0.4]  # reference [0.33, 0.67]
    report = run_evaluation(tmp_path, evidence)
    assert report["status"] == "failed"
    assert report["argmax_flips"]["dotnet-text"]["outside_margin"][0]["question"] == "three/q"
    assert any("argmax flips" in failure for failure in report["failures"])
    assert any("accuracy" not in failure for failure in report["failures"])


def test_flip_inside_margin_is_listed_not_failed(tmp_path):
    evidence = fake_evidence(tmp_path)
    for stage, key in (
        ("onnx", "output"),
        ("onnx", "uncached_output"),
        ("dotnet", "output"),
        ("dotnet", "uncached_output"),
        ("dotnet", "text_output"),
    ):
        evidence["documents"][stage]["cases"][2][key][2] = [0.49999, 0.50001]
    evidence["documents"]["onnx"]["cases"][2]["batch"]["output"][2] = [0.49999, 0.50001]
    report = run_evaluation(tmp_path, evidence)
    assert report["status"] == "passed", report["failures"]
    listed = report["argmax_flips"]["python-onnx-cached"]["listed_near_ties"]
    assert [entry["question"] for entry in listed] == ["three/r"]
    assert listed[0]["reference_margin"] < 1e-4
    assert "three/r (margin" in evaluate_decision.render_markdown(report)


def test_accuracy_must_be_identical(tmp_path):
    evidence = fake_evidence(tmp_path)
    # A labelled question that flips inside the near-tie margin still changes accuracy: identical is required.
    reference = json.loads(evidence["reference"].read_text(encoding="utf-8"))
    reference["cases"][0]["probs"][0] = [0.50002, 0.49998]
    evidence["reference"].write_text(json.dumps(reference), encoding="utf-8")
    digest = fixtures.sha256_file(evidence["reference"])
    for stage in ("onnx", "dotnet"):
        evidence["documents"][stage]["reference_sha256"] = digest
        for key in ("output", "uncached_output", "text_output"):
            if key in evidence["documents"][stage]["cases"][0]:
                evidence["documents"][stage]["cases"][0][key][0] = [0.49999, 0.50001]
    evidence["documents"]["onnx"]["cases"][0]["batch"]["output"][0] = [0.49999, 0.50001]
    report = run_evaluation(tmp_path, evidence)
    assert report["accuracy"]["status"] == "differs"
    assert report["status"] == "failed"


def test_token_id_mismatch_fails(tmp_path):
    evidence = fake_evidence(tmp_path)
    evidence["documents"]["dotnet"]["cases"][1]["rows"][0]["ids"][1] = 8
    evidence["documents"]["dotnet"]["cases"][0]["state_ids"].append(5)
    report = run_evaluation(tmp_path, evidence)
    assert report["status"] == "failed"
    assert report["token_ids"]["mismatches"] == ["two: state_ids", "one: rows"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", None])
def test_nan_or_non_numeric_fails(tmp_path, bad):
    evidence = fake_evidence(tmp_path)
    evidence["documents"]["dotnet"]["cases"][0]["text_output"][1][0] = bad
    report = run_evaluation(tmp_path, evidence)
    assert report["status"] == "failed"
    if isinstance(bad, float):
        assert report["nan_or_inf"] == ["dotnet-text/two/b"]
    else:
        assert any("non-numeric" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["dotnet"].update(cancellation_verified=False), "cancellation"),
        (lambda d: d["dotnet"]["environment"].update(machine="other"), "same named machine"),
        (lambda d: d["dotnet"].update(reference_sha256="0" * 64), "different reference.json"),
        (lambda d: d["onnx"]["environment"]["model_sha256"].update({"model.onnx": "f" * 64}), "built bundle"),
        (lambda d: d["dotnet"]["environment"].update(model_sha256={"model.onnx": "0" * 64}), "built bundle"),
        (lambda d: d["onnx"].update(candidate_sha256="0" * 64), "another candidate"),
        (lambda d: d["onnx"]["cases"][0].update(batch=None), "missing padded batch"),
    ],
)
def test_identity_and_evidence_failures(tmp_path, mutate, message):
    evidence = fake_evidence(tmp_path)
    mutate(evidence["documents"])
    report = run_evaluation(tmp_path, evidence)
    assert report["status"] == "failed"
    assert any(message in failure for failure in report["failures"]), report["failures"]


def test_missing_case_is_refused(tmp_path):
    evidence = fake_evidence(tmp_path)
    evidence["documents"]["dotnet"]["cases"].pop()
    with pytest.raises(PipelineError, match="missing or unknown cases"):
        run_evaluation(tmp_path, evidence)


def test_loosened_policy_is_refused(tmp_path):
    evidence = fake_evidence(tmp_path)
    evidence["candidate"]["evaluation"]["policy"]["padded_batch_vs_single_probability_max_abs"] = 1e-3
    with pytest.raises(PipelineError, match="predeclared"):
        run_evaluation(tmp_path, evidence)


# --- orchestration --------------------------------------------------------------------------------------------------


def staged(tmp_path: Path, monkeypatch) -> tuple[list, dict]:
    monkeypatch.setattr(
        onboarding_candidate,
        "environment_identity",
        lambda *args: {"python": {"status": "available"}, "dotnet": {"status": "available"}},
    )
    snapshot = tmp_path / "kev"
    snapshot.mkdir()
    candidate = profile()
    base = hf_base(tmp_path, candidate)
    for record in candidate["origin"]["files"]:
        (snapshot / record["path"]).write_bytes(b"x")
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    recipe = {
        "recipe_version": 1,
        "id": "kev-0.8b-cpu",
        "plugin": kev.PLUGIN,
        "manifest": "model-pipeline/candidates/kev-0.8b/profile.json",
        "reference_corpus": "model-pipeline/reference/corpus/kev-decisions-v1.jsonl",
        "reference_environment": "model-pipeline/reference/kev/requirements.txt",
    }
    stages = kev.build_stages(
        recipe, repo_root=REPO, snapshot=snapshot, reference_python=python, base_snapshot=base, threads=2
    )
    return stages, {"snapshot": snapshot, "base": base}


def test_stage_graph_declares_inputs_and_order(tmp_path, monkeypatch):
    stages, paths = staged(tmp_path, monkeypatch)
    assert [stage.name for stage in stages] == kev.STAGES
    graph = {stage.name: tuple(stage.dependencies) for stage in stages}
    assert graph["onnx"] == ("reference", "build", "environment")
    assert graph["evaluate"] == ("reference", "build", "onnx", "dotnet")
    assets = stages[0]
    assert assets.inputs["base_snapshot"] == paths["base"] and assets.inputs["snapshot"] == paths["snapshot"]
    reference = stages[2]
    assert "model-pipeline/reference/kev/run_kev.py" in reference.inputs
    assert "model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml" in reference.inputs
    assert "corpus" in reference.inputs
    assert all(stage.parameters["threads"] == 2 for stage in stages)


def test_build_stages_refuses_wrong_base_layout(tmp_path, monkeypatch):
    stages, paths = staged(tmp_path, monkeypatch)
    wrong = tmp_path / "flat"
    wrong.mkdir()
    recipe = {
        "recipe_version": 1,
        "plugin": kev.PLUGIN,
        "manifest": "model-pipeline/candidates/kev-0.8b/profile.json",
        "reference_corpus": "model-pipeline/reference/corpus/kev-decisions-v1.jsonl",
    }
    with pytest.raises(PipelineError, match="snapshots"):
        kev.build_stages(
            recipe,
            repo_root=REPO,
            snapshot=paths["snapshot"],
            reference_python=tmp_path / "python",
            base_snapshot=wrong,
        )


def test_evaluate_and_review_stages_with_fake_outputs(tmp_path, monkeypatch):
    stages, _ = staged(tmp_path, monkeypatch)
    by_name = {stage.name: stage for stage in stages}
    (tmp_path / "evidence").mkdir()
    evidence = fake_evidence(tmp_path / "evidence")
    # The stage computes the candidate digest from the shipped profile file.
    shipped = REPO / "model-pipeline/candidates/kev-0.8b/profile.json"
    digest = fixtures.sha256_file(shipped)
    reference = json.loads(evidence["reference"].read_text(encoding="utf-8"))
    reference["candidate_sha256"] = digest
    deps = {name: tmp_path / "deps" / name for name in ("assets", "reference", "build", "onnx", "dotnet")}
    for directory in deps.values():
        directory.mkdir(parents=True)
    (deps["reference"] / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    reference_digest = fixtures.sha256_file(deps["reference"] / "reference.json")
    for name in ("build", "onnx", "dotnet"):
        document = evidence["documents"][name]
        document["candidate_sha256"] = digest
        if name != "build":
            document["reference_sha256"] = reference_digest
        (deps[name] / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    (deps["assets"] / "source-lock.json").write_text(json.dumps({"profile": "kev"}), encoding="utf-8")
    out = tmp_path / "out" / "evaluate"
    result = by_name["evaluate"].execute(out, deps)
    assert result["status"] == "passed"
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# Decision development parity")
    review_out = tmp_path / "out" / "review"
    by_name["review"].execute(review_out, {**deps, "evaluate": out})
    review = (review_out / "review.md").read_text(encoding="utf-8")
    assert review.startswith("# Candidate review: kev-0.8b/onnx-fp32-cpu")
    assert "model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml" in review
    assert "| dotnet-text/vs_reference |" in review
    document = json.loads((review_out / "review.json").read_text(encoding="utf-8"))
    assert document["support_status"] == "experimental" and document["protected_fixture_adoption"] is False

    # A failing comparison fails the evaluate stage and leaves its report behind.
    bad = evidence["documents"]["dotnet"]
    bad["cases"][0]["output"][0] = [float("nan"), 0.5]
    (deps["dotnet"] / "dotnet.json").write_text(json.dumps(bad), encoding="utf-8")
    failing = tmp_path / "out" / "evaluate-failed"
    with pytest.raises(PipelineError, match="nan_or_inf"):
        by_name["evaluate"].execute(failing, deps)
    assert json.loads((failing / "report.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_dotnet_stage_reports_a_missing_tool(tmp_path, monkeypatch):
    stages, _ = staged(tmp_path, monkeypatch)
    dotnet = next(stage for stage in stages if stage.name == "dotnet")
    if (REPO / kev.TOOL).is_file():
        pytest.skip("the .NET fidelity tool exists in this checkout")
    with pytest.raises(PipelineError, match="tool is missing"):
        dotnet.execute(tmp_path / "out", {"build": tmp_path, "reference": tmp_path})
