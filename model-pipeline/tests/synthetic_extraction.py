"""A synthetic entity-extraction profile in a throwaway Git repository, for evaluator tests (P-015).

Four cases with max span width 3 and made-up digests. Span logits are chosen so each case has
known reference entities and known spans near the threshold:

    basic        "Ann met Bob in Rome."  person: Ann, Bob (confidences 1.6e-5 apart, a near tie), location: Rome;
                 location "in" just below the threshold (margin ~1e-5), person "met" far below it
    near         "Eve saw Kim."          person: Kim far above, Eve just above the threshold (margin ~1e-5)
    empty        ""                      normalized to ".", gate open, no entities
    gate-closed  "Nothing here."         count gate closed: no entities

Stage results are generated from the reference with controlled edits, so each test states exactly
which stage diverges and how.
"""

from __future__ import annotations

import copy
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

from model_pipeline import fixtures, results
from model_pipeline.evaluate_extraction import f32, sigmoid32
from model_pipeline.manifest import load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"
PROFILE = "synthetic-extractor/fp32-cpu"
PROTECTED = f"model-pipeline/protected/{PROFILE}"
GRAPH_SHA = "00000000000000000000000000000000000000000000000000000000000000b1"
DATA_SHA = "00000000000000000000000000000000000000000000000000000000000000b2"
TOKENIZER_SHA = "00000000000000000000000000000000000000000000000000000000000000b3"
K = 3
THRESHOLD = 0.5
DIM = 4
FAR = -10.0

TOLERANCES = {
    "profile": PROFILE,
    "reference": "synthetic",
    "discrete": {
        "normalized_text": "exact",
        "words": "exact",
        "character_maps": "exact",
        "token_ids": "exact",
        "attention_mask": "exact",
        "word_positions": "exact",
        "schema_positions": "exact",
        "gate_decision": "exact",
    },
    "float": {
        "last_hidden_state_max_abs": 1.0e-4,
        "count_logits_max_abs": 5.0e-4,
        "span_logits_relative": 1.0e-4,
        "scores_max_abs": 2.0e-5,
        "confidence_max_abs": 2.0e-5,
        "padded_batch_vs_single_span_logits_relative": 1.0e-4,
        "nan_or_inf": "fail",
        "report_only": ["mean_abs", "p99_abs"],
    },
    "entities": {
        "identity": "exact",
        "order_swap_max_confidence_delta": 2.0e-5,
        "threshold_flip_max_score_margin": 2.0e-5,
    },
}


def logit(p: float) -> float:
    return f32(math.log(p / (1 - p)))


def words_of(text: str) -> tuple[list[str], list[int], list[int]]:
    """Whitespace and single punctuation words, enough for synthetic text."""
    words, starts, ends, i = [], [], [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i + 1
        if text[i].isalnum():
            while j < len(text) and text[j].isalnum():
                j += 1
        words.append(text[i:j].lower())
        starts.append(i)
        ends.append(j)
        i = j
    return words, starts, ends


def build_case(cid: str, text: str, labels: list[str], spans: dict[tuple[str, int, int], float], gate: bool) -> dict:
    """`spans`: (label, start word, width) -> logit. Every other valid span gets logit FAR."""
    normalized = text if text[-1:] in (".", "!", "?") else text + "."
    words, starts, ends = words_of(normalized)
    n = len(words)
    schema = [1, 2] + [3] * len(labels)
    input_ids = [10, 11] + [12 + i for i in range(len(labels))] + [9] + [100 + i for i in range(n)]
    schema_index = [1] + [2 + i for i in range(len(labels))]
    word_index = [3 + len(labels) + i for i in range(n)]
    logits = []
    for label in labels:
        for s in range(n):
            for w in range(min(K, n - s)):
                logits.append(spans.get((label, s, w), FAR))
    entities: dict[str, list[dict[str, Any]]] = {}
    if gate:
        for label in labels:
            found = []
            for (lab, s, w), x in spans.items():
                score = sigmoid32(x)
                if lab == label and score >= f32(THRESHOLD):
                    found.append(
                        {
                            "text": normalized[starts[s] : ends[s + w]].strip(),
                            "confidence": score,
                            "start": starts[s],
                            "end": ends[s + w],
                        }
                    )
            entities[label] = sorted(found, key=lambda e: e["confidence"], reverse=True)
    count = [2.0, 5.0] + [-3.0] * 18 if gate else [5.0, 2.0] + [-3.0] * 18
    return {
        "id": cid,
        "category": "synthetic",
        "text": text,
        "labels_requested": labels,
        "threshold": THRESHOLD,
        "normalized_text": normalized,
        "words": words,
        "start_map": starts,
        "end_map": ends,
        "schema_tokens": schema,
        "labels": labels,
        "input_ids": input_ids,
        "word_index": word_index,
        "schema_index": schema_index,
        "count_logits": count,
        "pred_count": 1 if gate else 0,
        "gate_open": gate,
        "span_logits_shape": [len(labels), n, K],
        "span_logits_valid": logits,
        "output": {"entities": entities},
    }


CASES = [
    build_case(
        "basic",
        "Ann met Bob in Rome.",
        ["person", "location"],
        {
            ("person", 0, 0): logit(0.99999),  # Ann
            ("person", 2, 0): logit(0.999975),  # Bob, a near tie with Ann
            ("person", 1, 0): FAR,  # met
            ("location", 4, 0): logit(0.97),  # Rome
            ("location", 3, 0): logit(0.49999),  # in, just below the threshold
        },
        True,
    ),
    build_case(
        "near",
        "Eve saw Kim.",
        ["person"],
        {("person", 2, 0): logit(0.995), ("person", 0, 0): logit(0.50001)},
        True,
    ),
    build_case("empty", "", ["person"], {}, True),
    build_case("gate-closed", "Nothing here.", ["person", "location"], {}, False),
]
BY_ID = {c["id"]: c for c in CASES}


def flat_entities(case: dict) -> dict[str, Any]:
    items = [
        {"label": label, "text": e["text"], "start": e["start"], "end": e["end"], "confidence": e["confidence"]}
        for label, entries in case["output"]["entities"].items()
        for e in entries
    ]
    return {"gate_open": case["gate_open"], "entities": items}


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class SyntheticExtractionRepo:
    def __init__(self, root: Path, tolerances: dict[str, Any] | None = TOLERANCES, graph: bool = True):
        self.root = root
        self.protected = root / PROTECTED
        self.manifest = root / "model-pipeline" / "registry" / PROFILE / "manifest.yaml"
        self.results = root / "results"
        self.results.mkdir(parents=True)
        self.protected.mkdir(parents=True)
        fixtures.write_jsonl(self.protected / "extraction.jsonl", CASES)
        if tolerances is not None:
            (self.protected / "tolerances.yaml").write_text(yaml.safe_dump(tolerances), encoding="utf-8")
        else:
            (self.protected / "README.md").write_text("no tolerances yet\n", encoding="utf-8")
        self.graph = graph
        self.write_manifest()
        git(root, "init", "-q")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "synthetic protected evidence")

    def write_manifest(self, edit=None) -> None:
        m = load(SYNTHETIC)
        m["id"] = PROFILE
        m["behavior"].update(tasks=["extraction"], offsets_unit="unicode-scalars", truncation="error")
        files = [{"path": "tokenizer.json", "sha256": TOKENIZER_SHA, "size": 1}]
        if self.graph:
            files = [
                {"path": "onnx/model.onnx", "sha256": GRAPH_SHA, "size": 1},
                {"path": "onnx/model.onnx.data", "sha256": DATA_SHA, "size": 1},
                *files,
            ]
        m["origin"]["files"] = files
        m["qualification"]["fixtures"] = PROTECTED
        m["qualification"]["tolerances"] = f"{PROTECTED}/tolerances.yaml"
        if edit:
            edit(m)
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")

    # -- stage results ---------------------------------------------------------------------------

    def _base(self, stage: str, assets: list[dict[str, str]], machine: str = "box") -> dict:
        return results.new_results(
            profile=PROFILE,
            stage=stage,
            candidate=f"synthetic {stage}",
            producer={"name": "tests", "repository_commit": None},
            environment={"machine": machine, "provider": "cpu"},
            assets=assets,
            task="entity-extraction",
        )

    def _write(self, data: dict, name: str) -> Path:
        path = self.results / f"{name}{results.RESULTS_SUFFIX}"
        fixtures.write_json(path, data)
        return path

    GRAPH_ASSETS = [
        {"path": "onnx/model.onnx", "sha256": GRAPH_SHA},
        {"path": "onnx/model.onnx.data", "sha256": DATA_SHA},
    ]

    def tokenizer(self, edits: dict[str, dict[str, Any]] | None = None, drop: str | None = None) -> Path:
        data = self._base("tokenizer", [{"path": "tokenizer.json", "sha256": TOKENIZER_SHA}])
        for case in CASES:
            if case["id"] == drop:
                continue
            outs = {k: copy.deepcopy(case[k]) for k in results.ENTITY_EXTRACTION_OUTPUTS["tokenizer"] if k in case}
            outs["attention_mask"] = [1] * len(case["input_ids"])
            outs.update((edits or {}).get(case["id"], {}))
            data["cases"][case["id"]] = outs
        return self._write(data, "tokenizer")

    def _run_outputs(self, case: dict, logit_noise: float, rel_noise: float, count: list[float] | None) -> dict:
        logits = [f32(v + logit_noise + rel_noise * v) for v in case["span_logits_valid"]]
        return {
            "input_ids": case["input_ids"],
            "word_index": case["word_index"],
            "schema_index": case["schema_index"],
            "count_logits": count or list(case["count_logits"]),
            "span_logits": logits,
            "entities": flat_entities(case),
        }

    def _batches(self, data: dict, output: str, name: str, noise: float = 0.0) -> None:
        ids = [c for c in ("basic", "near", "empty") if c in data["cases"] and output in data["cases"][c]]
        values = []
        for c in ids:
            v = copy.deepcopy(data["cases"][c][output])
            if isinstance(v, list):
                v = [f32(x + noise) for x in v]
            values.append(v)
        data["batches"][name] = {"output": output, "case_ids": ids, "values": values}

    def export(
        self,
        logit_noise: float = 0.0,
        rel_noise: float = 0.0,
        hidden_noise: float = 0.0,
        batch_noise: float = 0.0,
        entities: dict[str, dict] | None = None,
        count: dict[str, list[float]] | None = None,
        drop: str | None = None,
        assets: list[dict[str, str]] | None = None,
        machine: str = "box",
    ) -> Path:
        data = self._base("export", self.GRAPH_ASSETS if assets is None else assets, machine)
        for case in CASES:
            if case["id"] == drop:
                continue
            outs = self._run_outputs(case, logit_noise, rel_noise, (count or {}).get(case["id"]))
            if entities and case["id"] in entities:
                outs["entities"] = entities[case["id"]]
            data["cases"][case["id"]] = outs
        self._batches(data, "span_logits", "padded", batch_noise)
        hidden = {c["id"]: [math.cos(j) for j in range(len(c["input_ids"]) * DIM)] for c in CASES if c["id"] != drop}
        data["tensors"]["last_hidden_state"] = results.write_tensor(
            self.results,
            "export-hidden.f32",
            [
                (c, (len(v) // DIM, DIM), [x + (hidden_noise if j == 0 else 0.0) for j, x in enumerate(v)])
                for c, v in hidden.items()
            ],
        )
        data["tensors"]["reference_last_hidden_state"] = results.write_tensor(
            self.results, "export-reference-hidden.f32", [(c, (len(v) // DIM, DIM), v) for c, v in hidden.items()]
        )
        return self._write(data, "export")

    def dotnet(
        self,
        logit_noise: float = 0.0,
        rel_noise: float = 0.0,
        entities: dict[str, dict] | None = None,
        text_entities: dict[str, dict] | None = None,
        drop: str | None = None,
        drop_text: str | None = None,
        batch_noise: float = 0.0,
        machine: str = "box",
    ) -> Path:
        data = self._base("dotnet", [*self.GRAPH_ASSETS, {"path": "tokenizer.json", "sha256": TOKENIZER_SHA}], machine)
        for case in CASES:
            if case["id"] == drop:
                continue
            outs = self._run_outputs(case, logit_noise, rel_noise, None)
            if entities and case["id"] in entities:
                outs["entities"] = entities[case["id"]]
            if case["id"] != drop_text:
                outs["text_entities"] = (text_entities or {}).get(case["id"], flat_entities(case))
            data["cases"][case["id"]] = outs
        self._batches(data, "span_logits", "padded", batch_noise)
        self._batches(data, "text_entities", "text")
        return self._write(data, "dotnet")


def entity_result(case_id: str, edit) -> dict[str, Any]:
    """The reference entity result of a case, changed by `edit(items)` (which may return a new list)."""
    result = flat_entities(BY_ID[case_id])
    items = copy.deepcopy(result["entities"])
    changed = edit(items)
    result["entities"] = items if changed is None else changed
    return result
