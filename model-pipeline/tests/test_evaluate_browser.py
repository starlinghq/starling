"""Browser profiles (follow-up plan W6): conversion of conformance runs, the browser checks of the staged evaluator,
the Preview task metric, and the manifest and catalog support for browser platforms. Synthetic data only."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from synthetic_profile import EMBEDDING, IDS, MODEL_SHA, PROFILE, PROTECTED, TOLERANCES, SyntheticRepo

from model_pipeline import browser_results, catalog, evaluate, fixtures, manifest, results
from model_pipeline import evaluate_task_quality as tq
from model_pipeline.cli import main

SMOKE = Path(__file__).parent / "fixtures" / "g3-smoke"

# The shape of the int8-embed browser tolerances: no hidden-state bound, no C#-vs-Python bound, a same-graph bound and
# the Preview task metric.
INT8_LIKE = {
    "profile": PROFILE,
    "reference": "synthetic",
    "discrete": {"token_ids": "exact", "attention_mask": "exact", "offsets": "exact"},
    "float": {
        "embedding_max_abs": 1.0e-2,
        "min_cosine": 0.9995,
        "padded_batch_vs_single_embedding_max_abs": 1.0e-5,
        "nan_or_inf": "fail",
        "report_only": ["mean_abs", "p99_abs"],
    },
    "browser_vs_native_same_graph": {"embedding_max_abs": 1.0e-5},
    "task_metric_before_preview": {"retrieval_top10_overlap_vs_fp32": 0.95},
}
# The shape of the fp32 browser tolerances: neither same-graph nor Preview key.
FP32_LIKE = {
    k: v for k, v in INT8_LIKE.items() if k not in ("browser_vs_native_same_graph", "task_metric_before_preview")
}
FP32_LIKE["float"] = {
    "embedding_max_abs": 1.0e-5,
    "padded_batch_vs_single_embedding_max_abs": 1.0e-5,
    "nan_or_inf": "fail",
    "report_only": ["mean_abs", "p99_abs", "min_cosine"],
}


def browser_manifest(m: dict) -> None:
    m["execution"]["backend"] = {"name": "onnxruntime-web", "version": "1.30.0", "binding": "Starlings.Runtime.OnnxWeb"}
    m["execution"]["graph"] = {"asset": {"path": "model.onnx", "sha256": MODEL_SHA, "size": 1}}
    m["compatibility"] = [
        {"os": "linux", "arch": "x64", "provider": "wasm", "browser": "chromium-153", "status": "experimental"}
    ]


class BrowserRepo(SyntheticRepo):
    def __init__(self, root: Path, tolerances: dict):
        super().__init__(root, tolerances)
        self.write_manifest(browser_manifest)
        self.runs = root / "runs"
        self.runs.mkdir()

    def native(self, noise: float = 0.0, model_sha: str = MODEL_SHA) -> Path:
        """Stage 2 of a browser profile: native ONNX Runtime on the same graph, no hidden-state tensors."""
        data = self._base("export", [{"path": "model.onnx", "sha256": model_sha}])
        for cid, ids in self.cases.items():
            data["cases"][cid] = {"input_ids": ids, "embedding": self.export_embedding(cid, noise)}
        data["batches"]["short-b8"] = {
            "output": "embedding",
            "case_ids": ["ascii", "long"],
            "values": [data["cases"][c]["embedding"] for c in ("ascii", "long")],
        }
        return self._write(data, "export")

    def run_file(
        self, noise: float = 0.0, text_noise: float = 0.0, model_sha: str = MODEL_SHA, error: str | None = None
    ) -> Path:
        """A run_conformance.py result file: browser embeddings = reference + `noise` on component 1."""

        def emb(cid: str, extra: float = 0.0) -> list[float]:
            return fixtures.float32_list(v + ((noise + extra) if i == 1 else 0.0) for i, v in enumerate(EMBEDDING[cid]))

        run = {
            "profile": PROFILE,
            "provider": "wasm",
            "runtime": ".NET 10",
            "onnxruntime_web": {"version": "1.30.0", "bundle": "wasm", "num_threads": 4},
            "load": {"model_sha256": model_sha},
            "e5_tokenizer": [{"id": c, "input_ids": ids, "attention_mask": [1] * len(ids)} for c, ids in IDS.items()],
            "embeddings": {c: emb(c) for c in IDS},
            "batch8": {c: emb(c) for c in ("ascii", "long")},
            "generator_embeddings": {c: emb(c, text_noise) for c in IDS},
        }
        if error:
            run = {"error": error}
        path = self.runs / f"run-{len(list(self.runs.iterdir()))}.json"
        path.write_text(json.dumps({"label": "aot", "browser_version": "153", "run": run}), encoding="utf-8")
        return path

    def convert(self, run: Path) -> dict[str, Path]:
        return browser_results.convert(run, self.manifest, self.protected, self.results, machine="box")

    def evaluate(self, **kw) -> evaluate.Evaluation:
        return evaluate.evaluate(self.manifest, self.results, repo_root=self.root, protected_base="HEAD", **kw)


def check(ev: evaluate.Evaluation, check_id: str) -> dict:
    return next(c for c in ev.report["checks"] if c["id"] == check_id)


@pytest.fixture
def int8(tmp_path):
    return BrowserRepo(tmp_path / "repo", INT8_LIKE)


def test_browser_profile_passes_g1_g2_with_preview_metric_not_evaluated(int8):
    int8.native(noise=0.0)
    int8.convert(int8.run_file(noise=2e-7))
    ev = int8.evaluate(require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, [c for c in ev.report["checks"] if c["status"] not in ("passed",)]
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "passed"
    same = check(ev, "dotnet.embedding.vs_native_same_graph")
    assert same["status"] == "passed" and same["bound"] == 1e-5
    assert same["metrics"]["max_abs"] == pytest.approx(2e-7, rel=0.2)
    # The hidden-state key is absent from the protected file: no export tensors, reported as not applicable.
    assert check(ev, "export.last_hidden_state")["status"] == "not-applicable"
    # The text path: the generator batch rows vs the reference, and vs the ids-path singles.
    assert check(ev, "dotnet.text_embedding")["status"] == "passed"
    assert check(ev, "dotnet.batch.generator-all-cases.vs_single")["status"] == "passed"
    # No C#-vs-Python check for a browser profile: the same-graph check takes its place.
    assert not any(c["id"] == "dotnet.embedding.vs_python" for c in ev.report["checks"])
    preview = check(ev, "preview.retrieval_top10_overlap_vs_fp32")
    assert preview["status"] == "not-run" and preview["gate"] == "Preview" and preview["bound"] == 0.95
    assert "P-030" in preview["detail"]
    assert ev.report["preview_requirements"] == [preview]
    assert "Preview" not in ev.report["manifest_gates"]
    assert not any(c["status"] == "not-run" and c["gate"] in ("G1", "G2") for c in ev.report["checks"])
    assert "Preview requirement" in evaluate.markdown(ev.report)


def test_browser_vs_native_same_graph_beyond_the_bound_fails_g2(int8):
    int8.native(noise=0.0)
    int8.convert(int8.run_file(noise=1e-4))  # within embedding_max_abs 1e-2, beyond the same-graph 1e-5
    ev = int8.evaluate()
    assert check(ev, "dotnet.embedding")["status"] == "passed"
    assert check(ev, "dotnet.embedding.vs_native_same_graph")["status"] == "failed"
    assert ev.report["gates"]["G2"]["status"] == "failed"
    assert ev.exit_code == evaluate.EXIT_FAILED


def test_same_graph_needs_the_same_graph_digest(int8):
    int8.native(model_sha="ff" * 32)
    int8.convert(int8.run_file())
    ev = int8.evaluate()
    assert check(ev, "dotnet.embedding.vs_native_same_graph")["status"] == "invalid"
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_same_graph_without_native_stage_is_not_run(int8):
    int8.convert(int8.run_file())
    ev = int8.evaluate()
    assert check(ev, "dotnet.embedding.vs_native_same_graph")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_text_path_padding_beyond_the_bound_fails(int8):
    int8.native()
    int8.convert(int8.run_file(text_noise=1e-4))
    ev = int8.evaluate()
    assert check(ev, "dotnet.batch.generator-all-cases.vs_single")["status"] == "failed"
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_fp32_browser_profile_reports_same_graph_drift_without_a_bound(tmp_path):
    repo = BrowserRepo(tmp_path / "repo", FP32_LIKE)
    repo.native()
    repo.convert(repo.run_file(noise=3e-7))
    ev = repo.evaluate(require_gates=("G1", "G2"))
    same = check(ev, "dotnet.embedding.vs_native_same_graph")
    assert same["status"] == "not-applicable" and same["bound"] is None
    assert same["metrics"]["max_abs"] > 0
    assert "preview_requirements" not in ev.report
    assert ev.exit_code == evaluate.EXIT_OK


def test_a_bound_hidden_state_key_is_never_skipped_for_a_browser_profile(tmp_path):
    tol = copy.deepcopy(INT8_LIKE)
    tol["float"]["last_hidden_state_max_abs"] = 1e-4
    repo = BrowserRepo(tmp_path / "repo", tol)
    repo.native()
    repo.convert(repo.run_file())
    ev = repo.evaluate()
    assert check(ev, "export.last_hidden_state")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_a_native_profile_without_the_csharp_bound_is_unchanged(tmp_path):
    tol = {k: v for k, v in TOLERANCES.items() if k != "csharp_vs_python_same_machine_and_provider"}
    repo = SyntheticRepo(tmp_path / "repo", tol)
    repo.tokenizer()
    repo.export()
    repo.dotnet()
    ev = evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base="HEAD")
    assert check(ev, "dotnet.embedding.vs_python")["status"] == "not-run"
    assert not any(c["id"] == "dotnet.embedding.vs_native_same_graph" for c in ev.report["checks"])
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_converter_refuses_failed_runs_and_other_graphs(int8):
    with pytest.raises(browser_results.BrowserResultsError, match="failed"):
        int8.convert(int8.run_file(error="WebGPU validation failed.\nmore"))
    with pytest.raises(browser_results.BrowserResultsError, match="SHA-256"):
        int8.convert(int8.run_file(model_sha="ee" * 32))
    native = SyntheticRepo(int8.root.parent / "native")
    with pytest.raises(browser_results.BrowserResultsError, match="not a browser profile"):
        browser_results.convert(int8.run_file(), native.manifest, native.protected, native.results)


def test_converter_output_is_valid_stage_results_without_thresholds(int8):
    paths = int8.convert(int8.run_file())
    tok, dot = results.load(paths["tokenizer"]), results.load(paths["dotnet"])
    assert tok.stage == "tokenizer" and set(tok.cases) == set(IDS)
    assert dot.batches[browser_results.TEXT_BATCH]["output"] == "text_embedding"
    assert dot.cases["long"]["input_ids"] == IDS["long"]
    assert dot.environment["provider"] == "wasm"


def test_cli_browser_results(int8, capsys):
    run = int8.run_file()
    args = ["browser-results", str(run), "--manifest", str(int8.manifest), "--fixtures", str(int8.protected)]
    assert main([*args, "--out", str(int8.results)]) == 0
    assert "dotnet" in capsys.readouterr().out
    assert main([*args, "--out", str(int8.results), "--repository-commit", "x"]) == 0
    bad = int8.run_file(error="boom")
    assert main(["browser-results", str(bad), *args[2:], "--out", str(int8.results)]) == 2


# The Preview metric in G3 ---------------------------------------------------------------------------


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        capture_output=True,
        check=True,
    )


def g3_repo(root: Path, budget_digest: str) -> Path:
    protected = root / PROTECTED
    protected.mkdir(parents=True)
    (protected / "tolerances.yaml").write_text(yaml.safe_dump(INT8_LIKE), encoding="utf-8")
    budget = {"profile": PROFILE, "retrieval": {"dataset_sha256": budget_digest, "minimum": {"ndcg@10": 0.5}}}
    (protected / "task-quality.yaml").write_text(yaml.safe_dump(budget), encoding="utf-8")
    m = manifest.load(Path(__file__).parent / "data" / "synthetic-manifest.yaml")
    m["id"] = PROFILE
    m["qualification"]["fixtures"] = PROTECTED
    m["qualification"]["tolerances"] = f"{PROTECTED}/tolerances.yaml"
    path = root / "model-pipeline" / "registry" / PROFILE / "manifest.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "synthetic")
    return path


def embeddings(path: Path, data: tq.RetrievalSet) -> Path:
    """Documents one-hot; each query points at its most relevant document."""
    docs = sorted(data.corpus)
    rows = [
        {"kind": "document", "id": d, "embedding": [1.0 if j == i else 0.0 for j in range(16)]}
        for i, d in enumerate(docs)
    ]
    for q, judged in sorted(data.qrels.items()):
        best = max(judged, key=lambda d: (judged[d], d))
        rows.append({"kind": "query", "id": q, "embedding": [1.0 if j == docs.index(best) else 0.0 for j in range(16)]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def preview(ev: tq.Evaluation) -> dict:
    return ev.report["preview_requirements"][0]


def test_g3_preview_overlap_pass_fail_and_not_evaluated(tmp_path):
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    man = g3_repo(tmp_path / "repo", tq.dataset_digest(data.files[:3]))
    root = tmp_path / "repo"
    fp32 = embeddings(tmp_path / "fp32.jsonl", data)

    def run(results_path, **kw):
        return tq.evaluate(
            man, "retrieval", SMOKE / "retrieval", results_path, repo_root=root, protected_base="HEAD", **kw
        )

    ev = run(fp32, baseline=fp32)
    assert preview(ev)["status"] == "passed" and preview(ev)["value"] == pytest.approx(1.0)
    assert ev.report["gates"]["G3"]["status"] == "passed"  # the Preview metric is reported beside G3, not in it
    assert ev.exit_code == tq.EXIT_OK
    assert "Preview requirement" in tq.markdown(ev.report)

    ev = run(fp32)  # no baseline
    assert preview(ev)["status"] == "not-run" and "--baseline" in preview(ev)["detail"]
    assert ev.exit_code != tq.EXIT_FAILED


def test_g3_preview_overlap_below_the_bound_fails(tmp_path, monkeypatch):
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    man = g3_repo(tmp_path / "repo", tq.dataset_digest(data.files[:3]))
    fp32 = embeddings(tmp_path / "fp32.jsonl", data)
    real = tq.top10_overlap

    def lower(candidate, baseline):
        out = real(candidate, baseline)
        out["top10_overlap_vs_baseline.mean"] = 0.9
        return out

    monkeypatch.setattr(tq, "top10_overlap", lower)
    ev = tq.evaluate(
        man, "retrieval", SMOKE / "retrieval", fp32, baseline=fp32, repo_root=tmp_path / "repo", protected_base="HEAD"
    )
    assert preview(ev)["status"] == "failed" and preview(ev)["value"] == 0.9
    assert ev.report["gates"]["G3"]["status"] == "passed"
    assert ev.exit_code == tq.EXIT_FAILED


def test_g3_preview_overlap_needs_the_budgeted_held_out_set(tmp_path):
    data = tq.load_retrieval_set(SMOKE / "retrieval")
    man = g3_repo(tmp_path / "repo", "ab" * 32)  # the budget names another set
    fp32 = embeddings(tmp_path / "fp32.jsonl", data)
    ev = tq.evaluate(
        man, "retrieval", SMOKE / "retrieval", fp32, baseline=fp32, repo_root=tmp_path / "repo", protected_base="HEAD"
    )
    assert preview(ev)["status"] == "not-run" and "P-030" in preview(ev)["detail"]


# Manifest schema and catalog ------------------------------------------------------------------------


def _schema_problems(tmp_path: Path, edit) -> list:
    m = manifest.load(Path(__file__).parent / "data" / "synthetic-manifest.yaml")
    browser_manifest(m)
    edit(m)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
    return manifest.validate_file(path)


def test_schema_browser_rows(tmp_path):
    assert _schema_problems(tmp_path, lambda m: None) == []
    assert _schema_problems(tmp_path, lambda m: m["compatibility"][0].pop("browser"))
    assert _schema_problems(tmp_path, lambda m: m["compatibility"][0].update(browser="netscape"))

    def cpu_with_browser(m):
        m["compatibility"] = [
            {"os": "linux", "arch": "x64", "provider": "cpu", "browser": "chromium", "status": "supported"}
        ]

    assert _schema_problems(tmp_path, cpu_with_browser)


def test_catalog_labels_browser_platforms_and_regression_note():
    row = {"os": "linux", "arch": "x64", "provider": "wasm", "browser": "chromium-153"}
    assert catalog.Platform.of(row).label == "linux-x64 wasm chromium-153"
    assert catalog.Platform.of({"os": "linux", "arch": "x64", "provider": "cpu"}).label == "linux-x64 cpu"
    readiness = {"ready": False, "gates_not_passed": ["G3"], "untested_capabilities": []}
    text = catalog.stable_readiness_text(readiness, scheduled_regression=False, browser=True)
    assert "browser regression" in text and "GPU" not in text
    gpu = catalog.stable_readiness_text(readiness, scheduled_regression=False)
    assert gpu == "gates not passed: G3; promotion also awaits a green per-release GPU regression run"
