"""Staged fidelity evaluator, task image-embedding, on a synthetic profile in a throwaway Git repository."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

from model_pipeline import evaluate, fixtures, results
from model_pipeline.cli import main
from model_pipeline.evaluate_image import read_npz_float32
from model_pipeline.manifest import load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"
PROFILE = "synthetic-image/fp32-cpu"
PROTECTED = f"model-pipeline/protected/{PROFILE}"
SHAS = {
    "onnx/vision.onnx": "00000000000000000000000000000000000000000000000000000000000000c1",
    "onnx/text-mm.onnx": "00000000000000000000000000000000000000000000000000000000000000c2",
    "onnx/text.onnx": "00000000000000000000000000000000000000000000000000000000000000c3",
    "tokenizer.json": "00000000000000000000000000000000000000000000000000000000000000c4",
}
TOLERANCES = {
    "profile": PROFILE,
    "provider_options": {},
    "discrete": {"token_ids": "exact", "resized_pixels": "exact", "pixel_values": "exact", "image_positions": "exact"},
    "float": {
        "soft_tokens_row_relative": 1.0e-3,
        "token_embeddings_row_relative": 1.0e-3,
        "embedding_max_abs": 1.0e-5,
        "min_cosine": 0.9999995,
        "padded_batch_vs_single_embedding_max_abs": 1.0e-5,
        "nan_or_inf": "fail",
        "report_only": ["mean_abs", "p99_abs", "per_value_relative"],
    },
    "decoder_skiasharp": {"decoded_rgb_vs_pil": "exact"},
}
SOFT = 2  # soft tokens per image, width 3 (stand-ins for 512)
WIDTH = 3


def digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def image(name: str) -> dict:
    return {
        "image": name,
        "rgb_sha256": digest(f"{name}/rgb"),
        "digests": {k: digest(f"{name}/{k}") for k in ("resized_uint8", "pixel_values", "positions")},
    }


ROWS = [
    {
        "id": "img-a",
        "input_ids": [2, 255999, 258880, 258880, 258882, 1],
        "embedding": [0.6, 0.8, 0.0],
        "images": [image("a")],
    },
    {
        "id": "mm-b",
        "input_ids": [2, 77, 255999, 258880, 258880, 258882, 1],
        "embedding": [0.0, 0.6, 0.8],
        "images": [image("b")],
    },
    {"id": "q-c", "input_ids": [2, 9, 1], "embedding": [1.0, 0.0, 0.0], "images": []},
    {
        "id": "gray16",
        "input_ids": [2, 255999, 258880, 258880, 258882, 1],
        "embedding": [0.0, 1.0, 0.0],
        "images": [image("g")],
    },
]
MODES = {"a": "RGB", "b": "L", "g": "I;16"}


def soft(row_id: str) -> list[float]:
    return [float(i + 1) + (10.0 if row_id == "mm-b" else 0.0) for i in range(SOFT * WIDTH)]


def tokens(row: dict) -> list[float]:
    return [float(j % 7) - 3.0 for j in range(len(row["input_ids"]) * WIDTH)]


def npy(values: list[float], shape: tuple[int, ...]) -> tuple[bytes, str]:
    payload = struct.pack(f"<{len(values)}f", *values)
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + ", ".join(str(s) for s in shape) + ",), }"
    header = header.ljust(117) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + payload, hashlib.sha256(
        payload
    ).hexdigest()


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=root,
        capture_output=True,
        check=True,
    )


class Repo:
    def __init__(self, root: Path, tolerances: dict | None = None):
        self.root = root
        self.protected = root / PROTECTED
        self.manifest = root / "model-pipeline" / "registry" / PROFILE / "manifest.yaml"
        self.results = root / "results"
        self.tensors = root / "reference-tensors"
        for d in (self.protected, self.results, self.tensors):
            d.mkdir(parents=True)
        fixtures.write_jsonl(self.protected / "image-embeddings.jsonl", ROWS)
        images = [{"id": k, "file": f"{k}.png", "mode": v} for k, v in MODES.items()]
        (self.protected / "images.json").write_text(json.dumps({"images": images}), encoding="utf-8")
        tensor_digests = {}
        for row in ROWS:
            arrays, entries = {}, {}
            for k in range(len(row["images"])):
                data, sha = npy(soft(row["id"]), (SOFT, WIDTH))
                arrays[f"image{k}_soft_tokens.npy"], entries[f"image{k}_soft_tokens"] = data, {"sha256": sha}
            data, sha = npy(tokens(row), (len(row["input_ids"]), WIDTH))
            arrays["token_embeddings.npy"], entries["token_embeddings"] = data, {"sha256": sha}
            with zipfile.ZipFile(self.tensors / f"{row['id']}.npz", "w") as z:
                for name, data in arrays.items():
                    z.writestr(name, data)
            tensor_digests[row["id"]] = entries
        (self.protected / "tensors.json").write_text(json.dumps(tensor_digests), encoding="utf-8")
        (self.protected / "tolerances.yaml").write_text(yaml.safe_dump(tolerances or TOLERANCES), encoding="utf-8")
        m = load(SYNTHETIC)
        m["id"] = PROFILE
        m["behavior"].update(tasks=["image-embedding"], truncation="error")
        m["origin"]["files"] = [{"path": p, "sha256": s, "size": 1} for p, s in SHAS.items()]
        m["qualification"]["fixtures"] = PROTECTED
        m["qualification"]["tolerances"] = f"{PROTECTED}/tolerances.yaml"
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
        git(root, "init", "-q")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "synthetic protected evidence")

    def _base(self, stage: str, assets: list[str], provider_options: dict | None = None) -> dict:
        return results.new_results(
            profile=PROFILE,
            stage=stage,
            candidate=f"synthetic {stage}",
            producer={"name": "tests", "repository_commit": None},
            environment={"machine": "box", "provider": "cpu", "provider_options": provider_options or {}},
            assets=[{"path": p, "sha256": SHAS[p]} for p in assets],
            task="image-embedding",
        )

    def stage1(self, edit=None, drop: str | None = None) -> None:
        data = self._base("tokenizer", ["tokenizer.json"])
        for row in ROWS:
            if row["id"] in ("gray16", drop):
                continue
            outs = {"input_ids": list(row["input_ids"])}
            for k, im in enumerate(row["images"]):
                outs[f"image{k}_rgb_sha256"] = im["rgb_sha256"]
                outs[f"image{k}_resized_sha256"] = im["digests"]["resized_uint8"]
                outs[f"image{k}_pixel_values_sha256"] = im["digests"]["pixel_values"]
                outs[f"image{k}_positions_sha256"] = im["digests"]["positions"]
            if edit:
                edit(row["id"], outs)
            data["cases"][row["id"]] = outs
        fixtures.write_json(self.results / "tokenizer.results.json", data)

    def stage3(
        self,
        noise: float = 0.0,
        soft_noise: float = 0.0,
        batch_noise: float = 0.0,
        drop: str | None = None,
        provider_options=None,
    ) -> None:
        data = self._base("dotnet", list(SHAS), provider_options)
        rows = [r for r in ROWS if r["id"] not in ("gray16", drop)]
        for row in rows:
            data["cases"][row["id"]] = {"embedding": [x + noise for x in row["embedding"]]}
        data["tensors"]["soft_tokens"] = results.write_tensor(
            self.results,
            "soft_tokens.f32",
            [
                (
                    f"{r['id']}/image{k}",
                    (SOFT, WIDTH),
                    [v + (soft_noise if j == 0 else 0.0) for j, v in enumerate(soft(r["id"]))],
                )
                for r in rows
                for k in range(len(r["images"]))
            ],
        )
        data["tensors"]["token_embeddings"] = results.write_tensor(
            self.results,
            "token_embeddings.f32",
            [(r["id"], (len(r["input_ids"]), WIDTH), tokens(r)) for r in rows if r["images"]],
        )
        ids = [r["id"] for r in rows]
        data["batches"]["mixed"] = {
            "output": "embedding",
            "case_ids": ids,
            "values": [[x + noise + batch_noise for x in r["embedding"]] for r in rows],
        }
        fixtures.write_json(self.results / "dotnet.results.json", data)

    def run(self, tensors: bool = True, **kw) -> evaluate.Evaluation:
        return evaluate.evaluate(
            self.manifest,
            self.results,
            repo_root=self.root,
            protected_base="HEAD",
            reference_tensors=self.tensors if tensors else None,
            **kw,
        )


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path / "repo")


def check(ev: evaluate.Evaluation, check_id: str) -> dict:
    return next(c for c in ev.report["checks"] if c["id"] == check_id)


def test_all_checks_pass_and_the_16_bit_row_is_excluded_by_contract(repo):
    repo.stage1()
    repo.stage3()
    ev = repo.run(require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, [
        c for c in ev.report["checks"] if c["status"] not in ("passed", "not-applicable")
    ]
    assert ev.report["task"] == "image-embedding"
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "passed"
    assert list(ev.report["excluded_by_contract"]) == ["gray16"]
    assert check(ev, "dotnet-ids.soft_tokens")["metrics"]["cases"] == 2
    assert check(ev, "dotnet-ids.token_embeddings")["metrics"]["cases"] == 2  # text-only rows have no image tokens
    assert check(ev, "export.not_applicable")["status"] == "not-applicable"


def test_without_reference_tensors_g2_cannot_pass(repo):
    repo.stage1()
    repo.stage3()
    ev = repo.run(tensors=False, require_gates=("G1", "G2"))
    assert check(ev, "dotnet-ids.soft_tokens")["status"] == "not-run"
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "not-run"
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED


def test_a_preprocessing_digest_mismatch_fails_stage_1(repo):
    def edit(row_id, outs):
        if row_id == "mm-b":
            outs["image0_resized_sha256"] = digest("other")

    repo.stage1(edit)
    repo.stage3()
    ev = repo.run()
    assert check(ev, "tokenizer.resized_sha256")["status"] == "failed"
    assert "mm-b/image0" in check(ev, "tokenizer.resized_sha256")["detail"]
    assert ev.report["first_divergence"] == "tokenizer"
    assert ev.exit_code == evaluate.EXIT_FAILED


def test_rendered_ids_must_be_exact_and_complete(repo):
    def edit(row_id, outs):
        if row_id == "img-a":
            outs["input_ids"] = outs["input_ids"][:-1]

    repo.stage1(edit, drop="q-c")
    repo.stage3()
    ev = repo.run()
    detail = check(ev, "tokenizer.token_ids")["detail"]
    assert "missing cases ['q-c']" in detail and "img-a" in detail


def test_embedding_soft_token_and_batch_bounds(repo):
    repo.stage1()
    repo.stage3(noise=2e-5, soft_noise=0.1, batch_noise=5e-5)
    ev = repo.run()
    assert check(ev, "dotnet-ids.embedding")["status"] == "failed"
    assert check(ev, "dotnet-ids.soft_tokens")["status"] == "failed"  # 0.1 / 16 > 1e-3
    assert check(ev, "dotnet-ids.batch.mixed")["status"] == "failed"
    assert check(ev, "dotnet-ids.min_cosine")["status"] in ("passed", "failed")
    assert ev.report["first_divergence"] == "dotnet-ids"


def test_a_missing_required_row_fails_but_the_excluded_row_may_be_absent(repo):
    repo.stage1()
    repo.stage3(drop="mm-b")
    ev = repo.run()
    assert "missing cases ['mm-b']" in check(ev, "dotnet-ids.embedding")["detail"]


def test_a_row_the_candidate_rejected_fails_with_its_reason(repo):
    # Starlings.Fidelity.Tool embeddinggemma2-images records a typed per-input rejection (here: the decoder's pixel
    # limit) as the row's single output in both stages and continues. Evidence, but never a pass.
    reason = "EmbeddingGemma2ImageTooLargeException: The image is 8000 x 10000 pixels; the limit is 50000000."
    repo.stage1()
    repo.stage3(drop="mm-b")
    for name in ("tokenizer.results.json", "dotnet.results.json"):
        path = repo.results / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cases"]["mm-b"] = {"rejected": reason}
        fixtures.write_json(path, data)
    ev = repo.run(require_gates=("G1", "G2"))
    for check_id in ("tokenizer.token_ids", "dotnet-ids.embedding"):
        c = check(ev, check_id)
        assert c["status"] == "failed"
        assert "missing cases ['mm-b']" in c["detail"]
        assert "rejected by the candidate" in c["detail"] and "EmbeddingGemma2ImageTooLargeException" in c["detail"]
    assert check(ev, "tokenizer.resized_sha256")["status"] == "failed"  # the rejected row's image digests are missing
    assert ev.report["gates"]["G1"]["status"] == "failed"
    assert ev.report["gates"]["G2"]["status"] != "passed"
    assert ev.exit_code == evaluate.EXIT_FAILED


def test_provider_options_must_match_the_profile(repo):
    repo.stage1()
    repo.stage3(provider_options={"use_tf32": "1"})
    tol = copy.deepcopy(TOLERANCES)
    tol["provider_options"] = {"use_tf32": 0}
    (repo.protected / "tolerances.yaml").write_text(yaml.safe_dump(tol), encoding="utf-8")
    git(repo.root, "commit", "-qam", "cuda-like profile")
    ev = repo.run()
    assert check(ev, "dotnet-ids.provider_options")["status"] == "failed"


def test_reference_tensors_are_checked_against_the_protected_digests(repo, tmp_path):
    other = tmp_path / "other-tensors"
    other.mkdir()
    for f in repo.tensors.iterdir():
        (other / f.name).write_bytes(f.read_bytes())
    data, _ = npy([9.0] * (SOFT * WIDTH), (SOFT, WIDTH))
    with zipfile.ZipFile(other / "img-a.npz", "w") as z:
        z.writestr("image0_soft_tokens.npy", data)
        z.writestr("token_embeddings.npy", npy(tokens(ROWS[0]), (6, WIDTH))[0])
    repo.stage1()
    repo.stage3()
    with pytest.raises(evaluate.EvaluationError, match="tensors.json"):
        evaluate.evaluate(
            repo.manifest, repo.results, repo_root=repo.root, protected_base="HEAD", reference_tensors=other
        )
    shape, values = read_npz_float32(
        repo.tensors / "img-a.npz",
        "image0_soft_tokens",
        json.loads((repo.protected / "tensors.json").read_text())["img-a"]["image0_soft_tokens"]["sha256"],
    )
    assert shape == [SOFT, WIDTH] and values == soft("img-a")


def test_cli_writes_image_reports(repo, tmp_path):
    repo.stage1()
    repo.stage3()
    out = tmp_path / "out"
    code = main(
        [
            "evaluate",
            str(repo.manifest),
            "--results",
            str(repo.results),
            "--out",
            str(out),
            "--repo-root",
            str(repo.root),
            "--protected-base",
            "HEAD",
            "--reference-tensors",
            str(repo.tensors),
            "--require-gates",
            "G1,G2",
        ]
    )
    assert code == evaluate.EXIT_OK
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "(image-embedding)" in md and "dotnet-ids.soft_tokens" in md
    assert json.loads((out / "report.json").read_text())["gates"]["G2"]["status"] == "passed"
