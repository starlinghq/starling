"""Task-quality evaluation (G3): retrieval and NER metrics, budget handling, and the SMOKE fixtures."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml

from model_pipeline import evaluate_task_quality as tq
from model_pipeline.cli import main
from model_pipeline.manifest import load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"
SMOKE = Path(__file__).parent / "fixtures" / "g3-smoke"
PROTECTED = "model-pipeline/protected/synthetic-encoder/fp32-cpu"


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        capture_output=True,
        check=True,
    )


def one_hot(i: int, dim: int = 16) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(dim)]


def perfect_embeddings(path: Path, data: tq.RetrievalSet, swap: dict[str, str] | None = None) -> Path:
    """Each query points at its most relevant document (or, with `swap`, at another one)."""
    docs = sorted(data.corpus)
    rows = [{"kind": "document", "id": d, "embedding": one_hot(i)} for i, d in enumerate(docs)]
    for q, judged in sorted(data.qrels.items()):
        best = max(judged, key=lambda d: (judged[d], d))
        target = (swap or {}).get(q, best)
        rows.append({"kind": "query", "id": q, "embedding": one_hot(docs.index(target))})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class Repo:
    def __init__(self, root: Path, tasks=("embedding",), budget: dict | None = None):
        self.root = root
        protected = root / PROTECTED
        protected.mkdir(parents=True)
        (protected / "tolerances.yaml").write_text("profile: synthetic-encoder/fp32-cpu\n", encoding="utf-8")
        if budget is not None:
            (protected / "task-quality.yaml").write_text(yaml.safe_dump(budget), encoding="utf-8")
        m = load(SYNTHETIC)
        m["behavior"]["tasks"] = list(tasks)
        m["qualification"]["fixtures"] = PROTECTED
        m["qualification"]["tolerances"] = f"{PROTECTED}/tolerances.yaml"
        self.manifest = root / "model-pipeline" / "registry" / "synthetic" / "manifest.yaml"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
        git(root, "init", "-q")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "synthetic")

    def run(self, task, dataset, results, **kw):
        return tq.evaluate(self.manifest, task, dataset, results, repo_root=self.root, protected_base="HEAD", **kw)


# Metrics ------------------------------------------------------------------------------------------


def test_retrieval_metrics_by_hand():
    qrels = {"q": {"a": 2, "b": 1, "c": 0}}
    m = tq.retrieval_metrics({"q": ["c", "b", "x", "a"]}, qrels)["metrics"]
    dcg = 0 / math.log2(2) + 1 / math.log2(3) + 0 + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg@10"] == pytest.approx(dcg / idcg)
    assert m["recall@1"] == 0 and m["recall@5"] == 1.0
    assert m["mrr@10"] == pytest.approx(0.5)


def test_top10_overlap():
    cand = {"q": [str(i) for i in range(12)]}
    base = {"q": [str(i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 9, 10)]}
    r = tq.top10_overlap(cand, base)
    assert r["top10_overlap_vs_baseline.mean"] == pytest.approx(0.9)
    assert r["top10_overlap_vs_baseline.min"] == pytest.approx(0.9)


def test_ner_exact_and_partial(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "id": "r",
                "text": "Ada Lovelace met Charles in London.",
                "labels": ["person", "location"],
                "entities": [
                    {"start": 0, "end": 12, "label": "person"},
                    {"start": 17, "end": 24, "label": "person"},
                    {"start": 28, "end": 34, "label": "location"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data = tq.load_ner_set(gold)
    preds = {"r": [(0, 3, "person"), (17, 24, "person"), (17, 24, "person"), (28, 34, "person")]}
    m = tq.ner_metrics(data, preds)
    assert m["metrics"]["exact.micro.precision"] == pytest.approx(1 / 4)
    assert m["metrics"]["exact.micro.recall"] == pytest.approx(1 / 3)
    assert m["metrics"]["partial.micro.recall"] == pytest.approx(2 / 3)  # duplicates match once
    assert m["per_label"]["exact"]["location"]["recall"] == 0
    assert m["metrics"]["exact.macro.f1"] == pytest.approx((2 * 0.25 * 0.5 / 0.75 + 0) / 2)


def test_ner_rejects_bad_offsets_and_labels(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"id": "r", "text": "abc", "labels": ["x"], "entities": [{"start": 0, "end": 9, "label": "x"}]})
        + "\n"
    )
    with pytest.raises(tq.EvaluationError, match="offsets"):
        tq.load_ner_set(gold)
    data_path = SMOKE / "ner" / "gold.jsonl"
    data = tq.load_ner_set(data_path)
    preds = tmp_path / "p.jsonl"
    preds.write_text(
        "".join(json.dumps({"id": r, "entities": [{"start": 0, "end": 1, "label": "nope"}]}) + "\n" for r in data.rows),
        encoding="utf-8",
    )
    with pytest.raises(tq.EvaluationError, match="label"):
        tq.load_predictions(preds, data)


# SMOKE fixtures and budgets -------------------------------------------------------------------------


def test_smoke_fixtures_are_labeled_and_valid():
    for sub in ("retrieval", "ner"):
        meta = json.loads((SMOKE / sub / "dataset.json").read_text(encoding="utf-8"))
        assert meta["status"].startswith("SMOKE")
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    assert len(data.queries) == 6
    assert len(tq.load_ner_set(SMOKE / "ner" / "gold.jsonl").rows) == 6


def test_retrieval_without_budget_is_not_a_gate_result(tmp_path):
    repo = Repo(tmp_path)
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    ev = repo.run("retrieval", SMOKE / "retrieval", perfect_embeddings(tmp_path / "e.jsonl", data))
    assert ev.report["metrics"]["mrr@10"] == pytest.approx(1.0)
    assert ev.report["gates"]["G3"]["status"] == "not-run"
    assert ev.report["gates"]["G3"]["reason"].startswith(tq.NO_BUDGET)
    assert ev.exit_code == tq.EXIT_GATE_NOT_PASSED


def test_retrieval_budget_pass_fail_and_dataset_mismatch(tmp_path):
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    digest = tq.dataset_digest(data.files[:3])
    budget = {
        "profile": "synthetic-encoder/fp32-cpu",
        "retrieval": {"dataset_sha256": digest, "minimum": {"ndcg@10": 0.9, "top10_overlap_vs_baseline.mean": 0.95}},
    }
    repo = Repo(tmp_path, budget=budget)
    good = perfect_embeddings(tmp_path / "good.jsonl", data)
    ev = repo.run("retrieval", SMOKE / "retrieval", good, baseline=good)
    assert ev.report["gates"]["G3"]["status"] == "passed", ev.report["gates"]
    assert ev.exit_code == 0
    bad = perfect_embeddings(tmp_path / "bad.jsonl", data, swap={"q-paris": "d-lyon", "q-bread": "d-cake"})
    ev = repo.run("retrieval", SMOKE / "retrieval", bad, baseline=good)
    assert ev.report["gates"]["G3"]["status"] == "failed"
    assert ev.exit_code == tq.EXIT_FAILED
    # Without the baseline the overlap metric is not computed: not a gate result.
    ev = repo.run("retrieval", SMOKE / "retrieval", good)
    assert ev.report["gates"]["G3"]["status"] == "not-run"
    # Another dataset: metrics computed, never a gate result.
    other = tmp_path / "other"
    other.mkdir()
    for name in tq.RETRIEVAL_FILES:
        (other / name).write_bytes((SMOKE / "retrieval" / name).read_bytes())
    with (other / "qrels.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "q-paris", "doc_id": "d-lyon", "relevance": 0}) + "\n")
    ev = repo.run("retrieval", other, good, baseline=good)
    assert ev.report["gates"]["G3"]["status"] == "not-run"
    assert "not the budgeted set" in ev.report["gates"]["G3"]["reason"]


def test_changed_budget_is_refused(tmp_path):
    repo = Repo(tmp_path, budget={"profile": "synthetic-encoder/fp32-cpu"})
    (tmp_path / PROTECTED / "task-quality.yaml").write_text("profile: synthetic-encoder/fp32-cpu\nner: {}\n")
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    ev = repo.run("retrieval", SMOKE / "retrieval", perfect_embeddings(tmp_path / "e.jsonl", data))
    assert ev.exit_code == tq.EXIT_PROTECTED


def test_embeddings_must_cover_the_dataset_and_carry_no_thresholds(tmp_path):
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    path = perfect_embeddings(tmp_path / "e.jsonl", data)
    lines = path.read_text(encoding="utf-8").splitlines()
    (tmp_path / "short.jsonl").write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(tq.EvaluationError, match="missing"):
        tq.load_embeddings(tmp_path / "short.jsonl", data)
    row = json.loads(lines[0]) | {"threshold": 0.5}
    (tmp_path / "t.jsonl").write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(tq.results.ToleranceOverrideError):
        tq.load_embeddings(tmp_path / "t.jsonl", data)


def test_ner_cli_and_task_mismatch(tmp_path):
    repo = Repo(tmp_path, tasks=("extraction",))
    data = tq.load_ner_set(SMOKE / "ner" / "gold.jsonl")
    preds = tmp_path / "p.jsonl"
    preds.write_text(
        "".join(
            json.dumps({"id": name, "entities": [{"start": s, "end": e, "label": lab} for s, e, lab in row["_gold"]]})
            + "\n"
            for name, row in data.rows.items()
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    code = main(
        [
            "evaluate",
            str(repo.manifest),
            "--g3",
            "ner",
            "--dataset",
            str(SMOKE / "ner" / "gold.jsonl"),
            "--results",
            str(preds),
            "--out",
            str(out),
            "--repo-root",
            str(tmp_path),
            "--protected-base",
            "HEAD",
        ]
    )
    assert code == tq.EXIT_GATE_NOT_PASSED
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["exact.micro.f1"] == 1.0
    assert "SMOKE" in (out / "report.md").read_text(encoding="utf-8")
    code = main(
        [
            "evaluate",
            str(repo.manifest),
            "--g3",
            "retrieval",
            "--dataset",
            str(SMOKE / "retrieval"),
            "--results",
            str(preds),
            "--out",
            str(out),
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert code == 2
