"""Task-quality evaluation (G3, plan section 10; workstream R5): retrieval, named-entity recognition, generation.

    model-pipeline evaluate <manifest> --g3 retrieval --dataset <dir> --results <embeddings.jsonl> --out <dir>
        [--baseline <fp32 embeddings.jsonl>]
    model-pipeline evaluate <manifest> --g3 ner --dataset <gold.jsonl> --results <predictions.jsonl> --out <dir>
    model-pipeline evaluate <manifest> --g3 generation --dataset <prompts.jsonl> --results <generations.jsonl | dir>
        --out <dir> [--baseline <reference generations.jsonl | dir>]

Metrics are computed from a dataset and the candidate's outputs; the comparison with a quality budget uses only the
maintainer-owned budget file `task-quality.yaml` in the profile's protected directory (the directory of the manifest's
`qualification.tolerances`). That file binds a budget to one dataset by digest: of the data files alone, or of the
data files followed by `dataset.json` when the dataset has one (both forms pin the same data bytes). Without the
file, or without an entry for the task, or when the dataset is not the one it names, the report says "no budget: not
a gate result" and G3 is `not-run` (exit 4). Choosing the G3 datasets and budgets is a maintainer decision
(production-readiness M1/M2); this module only defines formats and metrics. Dataset formats:
`model-pipeline/README.md`, "Task quality (G3)".

Budget file (protected; every value is a minimum, metric names as in the report):

    profile: multilingual-e5-small/onnx-fp32-cpu
    retrieval:
      dataset_sha256: <the report's dataset.sha256>
      minimum: {ndcg@10: 0.80, recall@100: 0.95, mrr@10: 0.70}
      # int8 browser profiles (kickoff decision 3): mean top-10 overlap with the fp32 results (--baseline)
      # minimum: {top10_overlap_vs_baseline.mean: 0.95}
    ner:
      dataset_sha256: <digest>
      minimum: {exact.micro.f1: 0.60, partial.micro.f1: 0.70}
    generation:
      dataset_sha256: <digest>
      minimum: {exact_match: 0.60}

Generation (exact match, for example GSM8K): the dataset is a prompts file in the format of
`reference/generation/prompts-v1.jsonl` whose rows also carry the reference `answer`. The results are one JSON line
per prompt with `id` and `output_text`: the output of `Starlings.Qwen35.Fidelity generation` (the .NET chat path:
owned template, greedy decoding), its `results.jsonl` from `--modes parity`, or the `results.jsonl` of the
transformers reference (`reference/generation/run_reference.py`). A directory is read as its `generations.jsonl`,
else its `results.jsonl`. The answer is the content of the last `\\boxed{...}` of the output (balanced braces), else
the last number in it. Normalization drops `$`, `,`, spaces, `%` and a trailing `.`, and unwraps `\\text{...}`. A
prediction matches when both sides parse as numbers equal within 1e-6, or the normalized strings are equal. This
is the metric of the Stable proposal (`reference/stable/score_generation.py`), fixed before any run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from model_pipeline import fixtures, manifest, protection, results
from model_pipeline.evaluate import (
    EXIT_FAILED,
    EXIT_GATE_NOT_PASSED,
    EXIT_OK,
    EXIT_PROTECTED,
    EvaluationError,
    _version,
)

REPORT_FORMAT = "starlings.task-quality-report"
REPORT_FORMAT_VERSION = 1
BUDGET_FILE = "task-quality.yaml"
RECALL_KS = (1, 5, 10, 100)
TASK_MANIFEST = {
    "retrieval": ("embedding", "image-embedding", "video-embedding"),
    "ner": ("extraction",),
    "generation": ("generation",),
}
NO_BUDGET = "no budget: not a gate result"
RETRIEVAL_FILES = ("queries.jsonl", "corpus.jsonl", "qrels.jsonl")


# --------------------------------------------------------------------------------------------
# Input readers


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise EvaluationError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path}: {exc}") from exc
    return rows


def _no_thresholds(path: Path, rows: Any) -> None:
    forbidden = results._find_forbidden(rows)
    if forbidden:
        raise results.ToleranceOverrideError(
            f"{path}: candidate outputs carry outputs only; budgets come from the protected {BUDGET_FILE}. "
            f"Rejected keys: {', '.join(forbidden)}"
        )


def _unique(rows: list[dict[str, Any]], path: Path, key: str = "id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get(key)
        if not isinstance(name, str) or not name or name in out:
            raise EvaluationError(f"{path}: missing, non-string or duplicate {key} {name!r}")
        out[name] = row
    return out


def dataset_digest(files: list[Path]) -> str:
    """SHA-256 over `<name>\\0<sha256>\\n` of each dataset file, in the given order."""
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.name}\0{fixtures.sha256_file(f)}\n".encode())
    return h.hexdigest()


@dataclass
class RetrievalSet:
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]
    files: list[Path]
    metadata: dict[str, Any] | None


def load_retrieval_set(directory: Path) -> RetrievalSet:
    if not directory.is_dir():
        raise EvaluationError(f"{directory}: a retrieval dataset is a directory with {', '.join(RETRIEVAL_FILES)}")
    files = [directory / name for name in RETRIEVAL_FILES]
    for f in files:
        if not f.is_file():
            raise EvaluationError(f"{directory}: missing {f.name}")
    queries = _unique(_jsonl(files[0]), files[0])
    corpus = _unique(_jsonl(files[1]), files[1])
    for path, rows in ((files[0], queries), (files[1], corpus)):
        for name, row in rows.items():
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise EvaluationError(f"{path}: {name!r} has no text")
    qrels: dict[str, dict[str, int]] = {}
    for row in _jsonl(files[2]):
        q, d, rel = row.get("query_id"), row.get("doc_id"), row.get("relevance")
        if q not in queries or d not in corpus:
            raise EvaluationError(f"{files[2]}: judgment {row} names an unknown query or document")
        if type(rel) is not int or rel < 0:
            raise EvaluationError(f"{files[2]}: relevance must be a non-negative integer: {row}")
        if d in qrels.setdefault(q, {}):
            raise EvaluationError(f"{files[2]}: duplicate judgment for {q!r}/{d!r}")
        qrels[q][d] = rel
    without = sorted(q for q in queries if not any(r > 0 for r in qrels.get(q, {}).values()))
    if without:
        raise EvaluationError(f"{files[2]}: queries without a relevant document (not evaluable): {without[:10]}")
    meta_path = directory / "dataset.json"
    metadata = None
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        files.append(meta_path)
    return RetrievalSet(
        {k: v["text"] for k, v in queries.items()}, {k: v["text"] for k, v in corpus.items()}, qrels, files, metadata
    )


@dataclass
class Embeddings:
    queries: dict[str, list[float]]
    documents: dict[str, list[float]]
    path: Path


def load_embeddings(path: Path, data: RetrievalSet) -> Embeddings:
    rows = _jsonl(path)
    if not rows:
        raise EvaluationError(f"{path}: no embeddings")
    _no_thresholds(path, rows)
    out: dict[str, dict[str, list[float]]] = {"query": {}, "document": {}}
    dim = None
    for row in rows:
        kind, name, vec = row.get("kind"), row.get("id"), row.get("embedding")
        if kind not in out or not isinstance(name, str):
            raise EvaluationError(f"{path}: row needs kind 'query' or 'document' and a string id: {str(row)[:120]}")
        if name in out[kind]:
            raise EvaluationError(f"{path}: duplicate {kind} {name!r}")
        if not isinstance(vec, list) or not vec or not all(isinstance(v, int | float) for v in vec):
            raise EvaluationError(f"{path}: {kind} {name!r} has no numeric embedding")
        if not all(math.isfinite(v) for v in vec):
            raise EvaluationError(f"{path}: {kind} {name!r} has a NaN or Inf value")
        if dim is None:
            dim = len(vec)
        elif len(vec) != dim:
            raise EvaluationError(f"{path}: {kind} {name!r} has dimension {len(vec)}, expected {dim}")
        out[kind][name] = [float(v) for v in vec]
    for kind, expected in (("query", data.queries), ("document", data.corpus)):
        missing = sorted(set(expected) - set(out[kind]))
        extra = sorted(set(out[kind]) - set(expected))
        if missing or extra:
            raise EvaluationError(
                f"{path}: {kind} embeddings missing {missing[:10]} or not in the dataset {extra[:10]}"
            )
    return Embeddings(out["query"], out["document"], path)


# --------------------------------------------------------------------------------------------
# Retrieval metrics


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def rank(emb: Embeddings) -> dict[str, list[str]]:
    """Per query, every document by descending cosine similarity; ties broken by document id."""
    docs = [(d, _normalize(v)) for d, v in sorted(emb.documents.items())]
    out = {}
    for q, v in emb.queries.items():
        qv = _normalize(v)
        scored = [(-sum(a * b for a, b in zip(qv, dv, strict=True)), d) for d, dv in docs]
        scored.sort()
        out[q] = [d for _, d in scored]
    return out


def retrieval_metrics(ranking: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    """nDCG@10 (linear gain rel, log2 discount; as trec_eval and BEIR), recall@k, MRR@10; means over queries."""
    per_query = {}
    for q, docs in ranking.items():
        judged = qrels[q]
        relevant = {d for d, r in judged.items() if r > 0}
        dcg = sum(judged.get(d, 0) / math.log2(i + 2) for i, d in enumerate(docs[:10]))
        ideal = sorted((r for r in judged.values() if r > 0), reverse=True)[:10]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        row = {"ndcg@10": dcg / idcg}
        for k in RECALL_KS:
            row[f"recall@{k}"] = len(relevant & set(docs[:k])) / len(relevant)
        rr = next((1.0 / (i + 1) for i, d in enumerate(docs[:10]) if d in relevant), 0.0)
        row["mrr@10"] = rr
        per_query[q] = row
    names = list(next(iter(per_query.values())))
    means = {m: sum(r[m] for r in per_query.values()) / len(per_query) for m in names}
    return {"metrics": means, "per_query": per_query}


def top10_overlap(candidate: dict[str, list[str]], baseline: dict[str, list[str]]) -> dict[str, Any]:
    """Per query |top10(candidate) & top10(baseline)| / min(10, corpus size): mean and minimum."""
    per = {}
    for q, docs in candidate.items():
        k = min(10, len(docs))
        per[q] = len(set(docs[:k]) & set(baseline[q][:k])) / k
    worst = min(per, key=lambda q: (per[q], q))
    return {
        "top10_overlap_vs_baseline.mean": sum(per.values()) / len(per),
        "top10_overlap_vs_baseline.min": per[worst],
        "worst_query": worst,
        "per_query": per,
    }


# --------------------------------------------------------------------------------------------
# NER


@dataclass
class NerSet:
    rows: dict[str, dict[str, Any]]
    labels: list[str]
    files: list[Path]


def _span(item: Any, text: str, labels: list[str], where: str) -> tuple[int, int, str]:
    if not isinstance(item, dict):
        raise EvaluationError(f"{where}: entity must be an object")
    s, e, label = item.get("start"), item.get("end"), item.get("label")
    if type(s) is not int or type(e) is not int or not 0 <= s < e <= len(text):
        raise EvaluationError(
            f"{where}: offsets [{s}, {e}) are not a non-empty span of the text ({len(text)} code points)"
        )
    if label not in labels:
        raise EvaluationError(f"{where}: label {label!r} is not in the row's labels {labels}")
    return s, e, label


def load_ner_set(path: Path) -> NerSet:
    if not path.is_file():
        raise EvaluationError(f"{path}: a NER dataset is one gold JSONL file")
    rows = _unique(_jsonl(path), path)
    if not rows:
        raise EvaluationError(f"{path}: empty")
    all_labels: list[str] = []
    for name, row in rows.items():
        text, labels = row.get("text"), row.get("labels")
        if not isinstance(text, str) or not text:
            raise EvaluationError(f"{path}: {name!r} has no text")
        if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
            raise EvaluationError(f"{path}: {name!r} needs a non-empty list of labels")
        if len(set(labels)) != len(labels):
            raise EvaluationError(f"{path}: {name!r} repeats a label")
        gold = row.get("entities")
        if not isinstance(gold, list):
            raise EvaluationError(f"{path}: {name!r} needs an entities list (may be empty)")
        spans = [_span(g, text, labels, f"{path}: {name!r}") for g in gold]
        if len(set(spans)) != len(spans):
            raise EvaluationError(f"{path}: {name!r} repeats a gold entity")
        row["_gold"] = spans
        all_labels += [x for x in labels if x not in all_labels]
    files = [path]
    meta = path.parent / "dataset.json"
    if meta.is_file():
        files.append(meta)
    return NerSet(rows, all_labels, files)


def load_predictions(path: Path, data: NerSet) -> dict[str, list[tuple[int, int, str]]]:
    rows = _jsonl(path)
    _no_thresholds(path, rows)
    preds = _unique(rows, path)
    missing = sorted(set(data.rows) - set(preds))
    extra = sorted(set(preds) - set(data.rows))
    if missing or extra:
        raise EvaluationError(f"{path}: predictions missing rows {missing[:10]} or not in the dataset {extra[:10]}")
    out = {}
    for name, row in preds.items():
        gold = data.rows[name]
        if not isinstance(row.get("entities"), list):
            raise EvaluationError(f"{path}: {name!r} needs an entities list")
        out[name] = [_span(p, gold["text"], gold["labels"], f"{path}: {name!r}") for p in row["entities"]]
    return out


def _match(gold: list[tuple[int, int, str]], pred: list[tuple[int, int, str]], partial: bool) -> list[tuple]:
    """One-to-one matching of predicted to gold spans of the same label (exact: identical offsets; partial: any
    overlap, largest overlap first, then document order). Returns matched (gold, pred) pairs."""
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g[2] != p[2]:
                continue
            if partial:
                overlap = min(g[1], p[1]) - max(g[0], p[0])
                if overlap > 0:
                    pairs.append((-overlap, gi, pi))
            elif g[:2] == p[:2]:
                pairs.append((0, gi, pi))
    pairs.sort()
    used_g, used_p, matched = set(), set(), []
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matched.append((gold[gi], pred[pi]))
    return matched


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, Any]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "predicted": n_pred, "gold": n_gold, "precision": p, "recall": r, "f1": f}


def ner_metrics(data: NerSet, preds: dict[str, list[tuple[int, int, str]]]) -> dict[str, Any]:
    """Span-level P/R/F1, exact and partial-overlap, per label, micro (pooled counts) and macro (mean over labels
    with at least one gold entity; labels without gold entities are listed, not averaged)."""
    out: dict[str, Any] = {"metrics": {}, "per_label": {}, "labels_without_gold": []}
    for variant, partial in (("exact", False), ("partial", True)):
        counts = {label: [0, 0, 0] for label in data.labels}  # tp, predicted, gold
        for name, row in data.rows.items():
            gold, pred = row["_gold"], preds[name]
            for g in gold:
                counts[g[2]][2] += 1
            for p in pred:
                counts[p[2]][1] += 1
            for g, _ in _match(gold, pred, partial):
                counts[g[2]][0] += 1
        per_label = {label: _prf(*c) for label, c in counts.items()}
        out["per_label"][variant] = per_label
        micro = _prf(*(sum(c[i] for c in counts.values()) for i in range(3)))
        scored = [label for label, c in counts.items() if c[2] > 0]
        macro = {
            m: sum(per_label[label][m] for label in scored) / len(scored) if scored else 0.0
            for m in ("precision", "recall", "f1")
        }
        for m in ("precision", "recall", "f1"):
            out["metrics"][f"{variant}.micro.{m}"] = micro[m]
            out["metrics"][f"{variant}.macro.{m}"] = macro[m]
            for label in data.labels:
                out["metrics"][f"{variant}.label.{label}.{m}"] = per_label[label][m]
        out["labels_without_gold"] = [label for label in data.labels if label not in scored]
    return out


# --------------------------------------------------------------------------------------------
# Generation (exact match)

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
GENERATION_RESULT_FILES = ("generations.jsonl", "results.jsonl")


@dataclass
class GenerationSet:
    rows: dict[str, dict[str, Any]]
    files: list[Path]


def load_generation_set(path: Path) -> GenerationSet:
    if not path.is_file():
        raise EvaluationError(f"{path}: a generation dataset is one prompts JSONL file with an answer per row")
    rows = _unique(_jsonl(path), path)
    if not rows:
        raise EvaluationError(f"{path}: empty")
    for name, row in rows.items():
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise EvaluationError(f"{path}: {name!r} needs a non-empty messages list")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise EvaluationError(f"{path}: {name!r} needs a non-empty string answer")
    files = [path]
    meta = path.parent / "dataset.json"
    if meta.is_file():
        files.append(meta)
    return GenerationSet(rows, files)


def generation_results_file(path: Path) -> Path:
    """A results file, or a runner's output directory (its generations.jsonl, else its results.jsonl)."""
    if path.is_dir():
        for name in GENERATION_RESULT_FILES:
            if (path / name).is_file():
                return path / name
        raise EvaluationError(f"{path}: no {' or '.join(GENERATION_RESULT_FILES)} in the directory")
    return path


def load_generations(path: Path, data: GenerationSet) -> dict[str, str]:
    rows = _jsonl(path)
    _no_thresholds(path, rows)
    outputs = _unique(rows, path)
    missing = sorted(set(data.rows) - set(outputs))
    extra = sorted(set(outputs) - set(data.rows))
    if missing or extra:
        raise EvaluationError(f"{path}: outputs missing rows {missing[:10]} or not in the dataset {extra[:10]}")
    out = {}
    for name, row in outputs.items():
        text = row.get("output_text")
        if not isinstance(text, str):
            raise EvaluationError(f"{path}: {name!r} needs output_text (the decoded generation)")
        out[name] = text
    return out


def last_boxed(text: str) -> str | None:
    """Content of the last `\\boxed{...}` (balanced braces); None without one, or when it is not closed."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
        i += 1
    return None


def extract_answer(text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    numbers = NUMBER.findall(text)
    return numbers[-1] if numbers else None


def normalize_answer(value: str) -> str:
    v = value.replace("$", "").replace(",", "").replace(" ", "").replace("\\%", "").replace("%", "")
    v = re.sub(r"^\\text\{(.*)\}$", r"\1", v)
    return v.rstrip(".")


def answer_matches(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    p, g = normalize_answer(prediction), normalize_answer(gold)
    try:
        return abs(float(p) - float(g)) <= 1e-6
    except ValueError:
        return p == g


def generation_metrics(data: GenerationSet, outputs: dict[str, str]) -> dict[str, Any]:
    """exact_match: the fraction of rows whose extracted answer matches the reference answer."""
    per_row = {}
    for name, row in data.rows.items():
        predicted = extract_answer(outputs[name])
        per_row[name] = {
            "answer": row["answer"],
            "extracted": predicted,
            "correct": answer_matches(predicted, row["answer"]),
        }
    correct = sum(r["correct"] for r in per_row.values())
    return {"metrics": {"exact_match": correct / len(per_row)}, "correct": correct, "per_row": per_row}


# --------------------------------------------------------------------------------------------
# Budget and report


@dataclass
class Evaluation:
    report: dict[str, Any]
    exit_code: int


def _budget(root: Path, man: dict[str, Any], task: str, protected_base: str):
    tol_rel = man["qualification"]["tolerances"]
    rel = str(Path(tol_rel).parent / BUDGET_FILE).replace("\\", "/")
    path = protection.resolve_protected(root, rel, kind="task-quality budget", must_exist=False)
    verification = protection.verify_unchanged(root, [path], protected_base)
    if not path.exists():
        return rel, None, verification, f"{NO_BUDGET} ({rel} does not exist; the maintainer sets it, M2)"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationError(f"{rel}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict) or data.get("profile") != man["id"]:
        raise protection.ProtectedEvidenceError(f"{rel} does not declare profile {man['id']!r}")
    entry = data.get(task)
    if not isinstance(entry, dict):
        return rel, None, verification, f"{NO_BUDGET} ({rel} has no '{task}' entry)"
    if not isinstance(entry.get("minimum"), dict) or not isinstance(entry.get("dataset_sha256"), str):
        raise EvaluationError(f"{rel}: '{task}' needs dataset_sha256 and a minimum mapping")
    return rel, entry, verification, None


def evaluate(
    manifest_path: Path,
    task: str,
    dataset: Path,
    results_path: Path,
    *,
    baseline: Path | None = None,
    repo_root: Path | None = None,
    protected_base: str = "auto",
    now: dt.datetime | None = None,
) -> Evaluation:
    problems = manifest.validate_file(manifest_path)
    if problems:
        raise EvaluationError("manifest is invalid: " + "; ".join(str(p) for p in problems))
    man = manifest.load(manifest_path)
    if not set(man["behavior"]["tasks"]) & set(TASK_MANIFEST[task]):
        raise EvaluationError(f"{man['id']}: task {task!r} needs a manifest task in {TASK_MANIFEST[task]}")
    root = repo_root or protection.find_repository_root(manifest_path.parent)

    extra: dict[str, Any] = {}
    if task == "retrieval":
        data = load_retrieval_set(dataset)
        digest = dataset_digest(data.files[:3])
        data_files = 3
        emb = load_embeddings(results_path, data)
        ranking = rank(emb)
        computed = retrieval_metrics(ranking, data.qrels)
        metrics = dict(computed["metrics"])
        extra["per_query"] = computed["per_query"]
        if baseline is not None:
            base_ranking = rank(load_embeddings(baseline, data))
            overlap = top10_overlap(ranking, base_ranking)
            metrics["top10_overlap_vs_baseline.mean"] = overlap["top10_overlap_vs_baseline.mean"]
            metrics["top10_overlap_vs_baseline.min"] = overlap["top10_overlap_vs_baseline.min"]
            extra["top10_overlap"] = {"worst_query": overlap["worst_query"], "per_query": overlap["per_query"]}
        sizes = {
            "queries": len(data.queries),
            "documents": len(data.corpus),
            "judgments": sum(map(len, data.qrels.values())),
        }
        meta = data.metadata
        files = data.files
    elif task == "ner":
        if baseline is not None:
            raise EvaluationError("--baseline applies to the retrieval and generation tasks only")
        data = load_ner_set(dataset)
        digest = dataset_digest(data.files[:1])
        data_files = 1
        computed = ner_metrics(data, load_predictions(results_path, data))
        metrics = computed["metrics"]
        extra["per_label"] = computed["per_label"]
        extra["labels_without_gold"] = computed["labels_without_gold"]
        sizes = {
            "rows": len(data.rows),
            "labels": len(data.labels),
            "gold_entities": sum(len(r["_gold"]) for r in data.rows.values()),
        }
        meta_path = dataset.parent / "dataset.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
        files = data.files
    elif task == "generation":
        data = load_generation_set(dataset)
        digest = dataset_digest(data.files[:1])
        data_files = 1
        results_path = generation_results_file(results_path)
        computed = generation_metrics(data, load_generations(results_path, data))
        metrics = dict(computed["metrics"])
        extra["per_row"] = computed["per_row"]
        if baseline is not None:
            baseline = generation_results_file(baseline)
            base = generation_metrics(data, load_generations(baseline, data))
            metrics["baseline_exact_match"] = base["metrics"]["exact_match"]
            extra["vs_baseline"] = {
                "baseline_exact_match": base["metrics"]["exact_match"],
                "flipped_rows": sorted(
                    k for k in data.rows if base["per_row"][k]["correct"] != computed["per_row"][k]["correct"]
                ),
            }
        sizes = {"rows": len(data.rows), "correct": computed["correct"]}
        meta_path = dataset.parent / "dataset.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
        files = data.files
    else:  # pragma: no cover - argparse restricts choices
        raise EvaluationError(f"unknown G3 task {task!r}")

    # The budget may name the digest of the data files alone (the documented form) or of the data files followed by
    # dataset.json (the form the accepted P-030 budgets were computed in). Both pin the same data bytes; the second
    # also pins the metadata.
    digest_with_metadata = dataset_digest(data.files) if len(data.files) > data_files else None
    budget_rel, entry, verification, no_budget = _budget(root, man, task, protected_base)
    checks = []
    matched_form = None
    gate_status, reason = "not-run", no_budget
    if entry is not None:
        for name, minimum in sorted(entry["minimum"].items()):
            value = metrics.get(name)
            if isinstance(minimum, bool) or not isinstance(minimum, int | float):
                checks.append(
                    {
                        "metric": name,
                        "minimum": minimum,
                        "value": value,
                        "status": "not-run",
                        "detail": "minimum is not a number",
                    }
                )
            elif value is None:
                checks.append(
                    {
                        "metric": name,
                        "minimum": minimum,
                        "value": None,
                        "status": "not-run",
                        "detail": "metric not computed (unknown name, or --baseline not given)",
                    }
                )
            else:
                checks.append(
                    {
                        "metric": name,
                        "minimum": minimum,
                        "value": value,
                        "status": "passed" if value >= minimum else "failed",
                        "detail": "",
                    }
                )
        matched_form = (
            "data"
            if entry["dataset_sha256"] == digest
            else "data+metadata"
            if digest_with_metadata is not None and entry["dataset_sha256"] == digest_with_metadata
            else None
        )
        if matched_form is None:
            gate_status = "not-run"
            reason = (
                f"{NO_BUDGET} (the dataset digest {digest[:16]} is not the budgeted set {entry['dataset_sha256'][:16]})"
            )
        elif verification.status == "changed":
            reason = f"{budget_rel} differs from the base commit; this run is not gate evidence"
        elif any(c["status"] == "failed" for c in checks):
            gate_status, reason = (
                "failed",
                "below budget: " + ", ".join(c["metric"] for c in checks if c["status"] == "failed"),
            )
        elif any(c["status"] == "not-run" for c in checks) or not checks:
            reason = "budget metrics not run: " + ", ".join(c["metric"] for c in checks if c["status"] == "not-run")
        elif verification.status != "verified":
            reason = f"budget not verified against Git ({verification.detail})"
        else:
            gate_status, reason = "passed", "every budgeted metric meets its minimum"

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p)

    date = (now or dt.datetime.now(dt.UTC)).date().isoformat()
    report = {
        "format": REPORT_FORMAT,
        "format_version": REPORT_FORMAT_VERSION,
        "task": task,
        "profile": man["id"],
        "date": date,
        "evaluator": {
            "name": "model-pipeline evaluate --g3",
            "version": _version(),
            "repository_commit": fixtures.git_head(root),
        },
        "dataset": {
            "sha256": digest,
            "sha256_with_metadata": digest_with_metadata,
            "files": [{"path": rel(f), "sha256": fixtures.sha256_file(f)} for f in files],
            "metadata": meta,
            "size": sizes,
        },
        "results": {"path": rel(results_path), "sha256": fixtures.sha256_file(results_path)},
        "baseline": None if baseline is None else {"path": rel(baseline), "sha256": fixtures.sha256_file(baseline)},
        "budget": {
            "path": budget_rel,
            "entry": entry,
            "status": "absent" if entry is None else "present",
            "matched_form": matched_form,
        },
        "protected_evidence": verification.as_dict(),
        "metrics": metrics,
        "budget_checks": checks,
        "gates": {"G3": {"status": gate_status, "reason": reason}},
        **extra,
    }
    if verification.status == "changed":
        code = EXIT_PROTECTED
    elif gate_status == "failed":
        code = EXIT_FAILED
    elif gate_status != "passed":
        code = EXIT_GATE_NOT_PASSED
    else:
        code = EXIT_OK
    report["exit_code"] = code
    return Evaluation(report, code)


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else ("" if value is None else str(value))


def markdown(report: dict[str, Any]) -> str:
    g3 = report["gates"]["G3"]
    meta = report["dataset"]["metadata"] or {}
    lines = [
        f"# Task quality (G3): {report['profile']} ({report['task']})",
        "",
        f"Date {report['date']}. Evaluator {report['evaluator']['name']} {report['evaluator']['version']} at "
        f"`{report['evaluator']['repository_commit']}`. Exit code {report['exit_code']}.",
        "",
        f"**G3: {g3['status']}**: {g3['reason']}.",
        "",
        f"Dataset `{report['dataset']['sha256']}`"
        + (f" ({meta.get('name')}, status {meta.get('status')})" if meta else "")
        + ": "
        + ", ".join(f"{k} {v}" for k, v in report["dataset"]["size"].items())
        + "."
        + (
            f" With `dataset.json`: `{report['dataset']['sha256_with_metadata']}`."
            if report["dataset"].get("sha256_with_metadata")
            else ""
        )
        + (
            f" The budget names the {report['budget']['matched_form']} digest."
            if report["budget"].get("matched_form")
            else ""
        ),
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in report["metrics"].items() if ".label." not in k]
    lines.append("")
    if report["task"] == "ner":
        lines += [
            "| Label | Exact P | Exact R | Exact F1 | Partial F1 | Gold | Predicted |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, e in report["per_label"]["exact"].items():
            p = report["per_label"]["partial"][label]
            lines.append(
                f"| {label} | {_fmt(e['precision'])} | {_fmt(e['recall'])} | {_fmt(e['f1'])} | "
                f"{_fmt(p['f1'])} | {e['gold']} | {e['predicted']} |"
            )
        lines.append("")
    if report["task"] == "generation" and report.get("vs_baseline"):
        flips = report["vs_baseline"]["flipped_rows"]
        lines += [f"Rows whose correctness differs from the baseline: {', '.join(flips) if flips else 'none'}.", ""]
    if report["budget_checks"]:
        lines += ["| Budgeted metric | Minimum | Value | Status |", "| --- | ---: | ---: | --- |"]
        lines += [
            f"| {c['metric']} | {_fmt(c['minimum'])} | {_fmt(c['value'])} | {c['status']} |"
            for c in report["budget_checks"]
        ]
        lines.append("")
    return "\n".join(lines)


def write_report(evaluation: Evaluation, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    fixtures.write_json(json_path, evaluation.report)
    md_path.write_text(markdown(evaluation.report), encoding="utf-8", newline="\n")
    return json_path, md_path
