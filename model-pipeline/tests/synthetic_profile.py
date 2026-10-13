"""A synthetic encoder profile in a throwaway Git repository, for evaluator tests.

Three cases, 4-dimensional embeddings, made-up digests. Stage results are generated from the
reference with controlled perturbations, so each test states exactly which stage diverges.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

from model_pipeline import fixtures, results
from model_pipeline.manifest import load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"
PROFILE = "synthetic-encoder/fp32-cpu"
PROTECTED = f"model-pipeline/protected/{PROFILE}"
MODEL_SHA = "00000000000000000000000000000000000000000000000000000000000000a1"
TOKENIZER_SHA = "00000000000000000000000000000000000000000000000000000000000000a2"
DIM = 4
IDS = {"ascii": [0, 5, 7, 2], "long": [0, 9, 3, 8, 6, 2], "empty": [0, 2]}

TOLERANCES = {
    "profile": PROFILE,
    "reference": "synthetic",
    "discrete": {"token_ids": "exact", "attention_mask": "exact", "offsets": "exact"},
    "float": {
        "last_hidden_state_max_abs": 1.0e-4,
        "embedding_max_abs": 1.0e-5,
        "padded_batch_vs_single_embedding_max_abs": 1.0e-5,
        "nan_or_inf": "fail",
        "report_only": ["mean_abs", "p99_abs", "min_cosine"],
    },
    "csharp_vs_python_same_machine_and_provider": {"embedding_max_abs": 1.0e-6},
}


def _vector(seed: int) -> list[float]:
    raw = [math.sin(seed * 1.7 + i) for i in range(DIM)]
    norm = math.sqrt(sum(v * v for v in raw))
    return fixtures.float32_list(v / norm for v in raw)


# Rows computed with a named prompt (EmbeddingGemma 2 style): case "<id>@<prompt>", ids of prompt + text.
PROMPTED = {"ascii@query": [0, 11, 12, 5, 7, 2], "empty@query": [0, 11, 12, 2]}
EMBEDDING = {cid: _vector(n + 1) for n, cid in enumerate(IDS | PROMPTED)}
HIDDEN = {
    cid: [math.cos(n + j) for j in range(len(ids) * DIM)] for n, (cid, ids) in enumerate((IDS | PROMPTED).items())
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class SyntheticRepo:
    def __init__(self, root: Path, tolerances: dict[str, Any] | None = None, prompted: bool = False):
        self.root = root
        self.cases = IDS | PROMPTED if prompted else dict(IDS)
        self.protected = root / PROTECTED
        self.manifest = root / "model-pipeline" / "registry" / PROFILE / "manifest.yaml"
        self.results = root / "results"
        self.results.mkdir(parents=True)
        self.protected.mkdir(parents=True)
        fixtures.write_jsonl(
            self.protected / "tokenizer.jsonl",
            [
                {
                    "id": cid,
                    "category": "x",
                    "text": f"text {cid}",
                    "encoding": {"ids": ids, "attention_mask": [1] * len(ids)},
                }
                for cid, ids in IDS.items()
            ]
            + [
                {
                    "id": "pair",
                    "category": "pair",
                    "text": "a",
                    "pair": "b",
                    "encoding": {"ids": [0, 2, 2], "attention_mask": [1, 1, 1]},
                }
            ],
        )
        fixtures.write_jsonl(
            self.protected / "embeddings.jsonl",
            [{"id": cid, "input_ids": IDS[cid], "embedding": EMBEDDING[cid]} for cid in IDS]
            + [
                {"id": cid.split("@")[0], "prompt": cid.split("@")[1], "input_ids": ids, "embedding": EMBEDDING[cid]}
                for cid, ids in self.cases.items()
                if "@" in cid
            ],
        )
        (self.protected / "tolerances.yaml").write_text(yaml.safe_dump(tolerances or TOLERANCES), encoding="utf-8")
        self.write_manifest()
        git(root, "init", "-q")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "synthetic protected evidence")

    def write_manifest(self, edit=None) -> None:
        m = load(SYNTHETIC)
        m["id"] = PROFILE
        m["origin"]["files"] = [
            {"path": "model.onnx", "sha256": MODEL_SHA, "size": 1},
            {"path": "tokenizer.json", "sha256": TOKENIZER_SHA, "size": 1},
        ]
        m["qualification"]["fixtures"] = PROTECTED
        m["qualification"]["tolerances"] = f"{PROTECTED}/tolerances.yaml"
        if edit:
            edit(m)
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")

    # -- stage results ---------------------------------------------------------------------------

    def _base(self, stage: str, assets: list[dict[str, str]], machine: str = "box", options=None) -> dict:
        env = {"machine": machine, "provider": "cpu"}
        if options is not None:
            env["provider_options"] = options
        return results.new_results(
            profile=PROFILE,
            stage=stage,
            candidate=f"synthetic {stage}",
            producer={"name": "tests", "repository_commit": None},
            environment=env,
            assets=assets,
        )

    def _write(self, data: dict, name: str) -> Path:
        path = self.results / f"{name}{results.RESULTS_SUFFIX}"
        fixtures.write_json(path, data)
        return path

    def tokenizer(self, ids: dict[str, list[int]] | None = None, drop: str | None = None) -> Path:
        data = self._base("tokenizer", [{"path": "tokenizer.json", "sha256": TOKENIZER_SHA}])
        for cid, ref in self.cases.items():
            if cid == drop:
                continue
            got = (ids or {}).get(cid, ref)
            data["cases"][cid] = {"input_ids": got, "attention_mask": [1] * len(got)}
        return self._write(data, "tokenizer")

    def export_embedding(self, cid: str, noise: float = 0.0) -> list[float]:
        return fixtures.float32_list(v + (noise if i == 0 else 0.0) for i, v in enumerate(EMBEDDING[cid]))

    def export(self, emb_noise: float = 0.0, hidden_noise: float = 0.0, batch_noise: float = 0.0, **kw) -> Path:
        data = self._base("export", [{"path": "model.onnx", "sha256": kw.get("model_sha", MODEL_SHA)}], **_env(kw))
        for cid in self.cases:
            data["cases"][cid] = {"input_ids": self.cases[cid], "embedding": self.export_embedding(cid, emb_noise)}
        data["batches"]["fixture"] = _batch("embedding", data, batch_noise)
        data["tensors"]["last_hidden_state"] = results.write_tensor(
            self.results,
            "export-hidden.f32",
            [
                (c, (len(self.cases[c]), DIM), [v + (hidden_noise if j == 0 else 0) for j, v in enumerate(HIDDEN[c])])
                for c in self.cases
            ],
        )
        data["tensors"]["reference_last_hidden_state"] = results.write_tensor(
            self.results,
            "export-reference-hidden.f32",
            [(c, (len(self.cases[c]), DIM), HIDDEN[c]) for c in self.cases],
        )
        return self._write(data, "export")

    def dotnet(
        self,
        export_noise: float = 0.0,
        noise: float = 0.0,
        text_noise: float = 0.0,
        batch_noise: float = 0.0,
        nan: bool = False,
        drop: str | None = None,
        ids: dict[str, list[int]] | None = None,
        **kw,
    ) -> Path:
        """C# outputs: Python ORT outputs (`export_noise`) plus a .NET-specific deviation (`noise`)."""
        data = self._base(
            "dotnet",
            [{"path": "model.onnx", "sha256": MODEL_SHA}, {"path": "tokenizer.json", "sha256": TOKENIZER_SHA}],
            **_env(kw),
        )
        for cid in self.cases:
            if cid == drop:
                continue
            emb = self.export_embedding(cid, export_noise)
            emb = fixtures.float32_list(v + (noise if i == 1 else 0.0) for i, v in enumerate(emb))
            if nan and cid == "long":
                emb[2] = float("nan")
            text = fixtures.float32_list(v + (text_noise if i == 2 else 0.0) for i, v in enumerate(emb))
            data["cases"][cid] = {
                "input_ids": (ids or {}).get(cid, self.cases[cid]),
                "embedding": emb,
                "text_embedding": text,
            }
        data["batches"]["fixture"] = _batch("embedding", data, batch_noise)
        data["batches"]["text"] = _batch("text_embedding", data, 0.0)
        return self._write(data, "dotnet")


def _env(kw: dict) -> dict:
    return {k: kw[k] for k in ("machine", "options") if k in kw}


def _batch(output: str, data: dict, noise: float) -> dict:
    ids = list(data["cases"])
    return {
        "output": output,
        "case_ids": ids,
        "values": [[v + (noise if i == 3 else 0.0) for i, v in enumerate(data["cases"][c][output])] for c in ids],
    }
