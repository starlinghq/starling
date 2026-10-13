"""Staged fidelity evaluator, task video-embedding, on a synthetic profile in a throwaway Git repository."""

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
from model_pipeline.manifest import load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"
PROFILE = "synthetic-video/fp32-cpu"
PROTECTED = f"model-pipeline/protected/{PROFILE}"
SHAS = {
    "onnx/vision.onnx": "00000000000000000000000000000000000000000000000000000000000000d1",
    "onnx/text-mm.onnx": "00000000000000000000000000000000000000000000000000000000000000d2",
    "onnx/text.onnx": "00000000000000000000000000000000000000000000000000000000000000d3",
    "tokenizer.json": "00000000000000000000000000000000000000000000000000000000000000d4",
}
TOLERANCES = {
    "profile": PROFILE,
    "provider_options": {},
    "discrete": {
        "frame_indices": "exact",
        "token_ids": "exact",
        "decoded_frames": "exact",
        "resized_pixels": "exact",
        "pixel_values": "exact",
        "video_positions": "exact",
    },
    "float": {
        "soft_tokens_row_relative": 1.0e-3,
        "token_embeddings_row_relative": 1.0e-3,
        "embedding_max_abs": 1.0e-5,
        "min_cosine": 0.9999995,
        "padded_batch_vs_single_embedding_max_abs": 1.0e-5,
        "nan_or_inf": "fail",
        "report_only": ["mean_abs", "p99_abs", "per_value_relative"],
    },
    "decoder_ffmpeg": {"decoded_rgb_vs_torchcodec": "exact", "supported_ffmpeg_major": 8},
}
FRAMES = 2  # sampled frames per video
SOFT = 1  # soft tokens per frame
WIDTH = 3  # stand-in for 512 / 768


def digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def video(name: str, indices: list[int]) -> dict:
    return {
        "video": name,
        "input_form": "path",
        "sampled_indices": indices,
        "frames": len(indices),
        "soft_tokens_per_frame": SOFT,
        "frame_sha256": [digest(f"{name}/frame{i}") for i in indices],
        "digests": {k: digest(f"{name}/{k}") for k in ("frames_uint8", "resized_uint8", "pixel_values", "positions")},
    }


VIDEO_IDS = [2, 255999, 258884, 258882, 255999, 258884, 258882, 1]
ROWS = [
    {
        "id": "vid-a",
        "input_ids": VIDEO_IDS,
        "embedding": [0.6, 0.8, 0.0],
        "image": None,
        "videos": [video("a", [0, 30])],
    },
    {
        "id": "mm-b",
        "input_ids": [2, 77, 255999, 258880, 258882, *VIDEO_IDS[1:]],
        "embedding": [0.0, 0.6, 0.8],
        "image": {"video": "b", "frame_index": 10},
        "videos": [video("b", [0, 25])],
    },
    {"id": "q-c", "input_ids": [2, 9, 1], "embedding": [1.0, 0.0, 0.0], "image": None, "videos": []},
    {
        "id": "vid-array-no-metadata",
        "input_ids": VIDEO_IDS,
        "embedding": [0.0, 1.0, 0.0],
        "image": None,
        "videos": [{**video("a", [0, 149]), "input_form": "array"}],
    },
]


def soft(row_id: str) -> list[float]:
    return [float(i + 1) + (10.0 if row_id == "mm-b" else 0.0) for i in range(FRAMES * SOFT * WIDTH)]


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
        fixtures.write_jsonl(self.protected / "video-embeddings.jsonl", ROWS)
        (self.protected / "videos.json").write_text(json.dumps({"videos": [{"id": "a"}, {"id": "b"}]}), "utf-8")
        tensor_digests = {}
        for row in ROWS:
            arrays, entries = {}, {}
            for k in range(len(row["videos"])):
                data, sha = npy(soft(row["id"]), (FRAMES * SOFT, WIDTH))
                arrays[f"video{k}_soft_tokens.npy"], entries[f"video{k}_soft_tokens"] = data, {"sha256": sha}
            data, sha = npy(tokens(row), (len(row["input_ids"]), WIDTH))
            arrays["token_embeddings.npy"], entries["token_embeddings"] = data, {"sha256": sha}
            if row["image"]:
                entries["image0_rgb"] = {"sha256": digest(f"{row['id']}/still")}
            with zipfile.ZipFile(self.tensors / f"{row['id']}.npz", "w") as z:
                for name, data in arrays.items():
                    z.writestr(name, data)
            tensor_digests[row["id"]] = entries
        (self.protected / "tensors.json").write_text(json.dumps(tensor_digests), encoding="utf-8")
        (self.protected / "tolerances.yaml").write_text(yaml.safe_dump(tolerances or TOLERANCES), encoding="utf-8")
        m = load(SYNTHETIC)
        m["id"] = PROFILE
        m["behavior"].update(tasks=["video-embedding"], truncation="error")
        m["origin"]["files"] = [{"path": p, "sha256": s, "size": 1} for p, s in SHAS.items()]
        m["qualification"]["fixtures"] = PROTECTED
        m["qualification"]["tolerances"] = f"{PROTECTED}/tolerances.yaml"
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
        git(root, "init", "-q")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "synthetic protected evidence")

    def _base(self, stage: str, assets: list[str], environment: dict) -> dict:
        return results.new_results(
            profile=PROFILE,
            stage=stage,
            candidate=f"synthetic {stage}",
            producer={"name": "tests", "repository_commit": None},
            environment={"machine": "box", "provider": "cpu", "provider_options": {}, **environment},
            assets=[{"path": p, "sha256": SHAS[p]} for p in assets],
            task="video-embedding",
        )

    def stage1(self, edit=None, drop: str | None = None, include_excluded=False, ffmpeg_major=8) -> None:
        env = {"ffmpeg_version": f"n{ffmpeg_major}.1.3", "ffmpeg_major": ffmpeg_major}
        if ffmpeg_major is None:
            env = {}
        data = self._base("tokenizer", ["tokenizer.json"], env)
        for row in ROWS:
            if row["id"] == drop or (row["id"] == "vid-array-no-metadata" and not include_excluded):
                continue
            outs = {"input_ids": list(row["input_ids"])}
            for k, v in enumerate(row["videos"]):
                outs[f"video{k}_frame_indices"] = list(v["sampled_indices"])
                outs[f"video{k}_frame_sha256"] = list(v["frame_sha256"])
                outs[f"video{k}_frames_sha256"] = v["digests"]["frames_uint8"]
                outs[f"video{k}_resized_sha256"] = v["digests"]["resized_uint8"]
                outs[f"video{k}_pixel_values_sha256"] = v["digests"]["pixel_values"]
                outs[f"video{k}_positions_sha256"] = v["digests"]["positions"]
            if row["image"]:
                outs["image0_rgb_sha256"] = digest(f"{row['id']}/still")
            if edit:
                edit(row["id"], outs)
            data["cases"][row["id"]] = outs
        fixtures.write_json(self.results / "tokenizer.results.json", data)

    def stage3(self, noise=0.0, soft_noise=0.0, batch_noise=0.0, drop=None, include_excluded=False) -> None:
        data = self._base("dotnet", list(SHAS), {})
        rows = [r for r in ROWS if r["id"] != drop and (include_excluded or r["id"] != "vid-array-no-metadata")]
        for row in rows:
            data["cases"][row["id"]] = {"embedding": [x + noise for x in row["embedding"]]}
        data["tensors"]["soft_tokens"] = results.write_tensor(
            self.results,
            "soft_tokens.f32",
            [
                (
                    f"{r['id']}/video{k}",
                    (FRAMES * SOFT, WIDTH),
                    [v + (soft_noise if j == 0 else 0.0) for j, v in enumerate(soft(r["id"]))],
                )
                for r in rows
                for k in range(len(r["videos"]))
            ],
        )
        data["tensors"]["token_embeddings"] = results.write_tensor(
            self.results,
            "token_embeddings.f32",
            [(r["id"], (len(r["input_ids"]), WIDTH), tokens(r)) for r in rows if r["videos"] or r["image"]],
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


def test_all_checks_pass_and_the_metadata_less_row_is_excluded(repo):
    repo.stage1()
    repo.stage3()
    ev = repo.run(require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, [
        c for c in ev.report["checks"] if c["status"] not in ("passed", "not-applicable")
    ]
    assert ev.report["task"] == "video-embedding"
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "passed"
    assert list(ev.report["excluded_by_contract"]) == ["vid-array-no-metadata"]
    assert ev.report["evaluated_although_excluded"] == []
    assert check(ev, "dotnet-ids.soft_tokens")["metrics"]["cases"] == 2
    assert check(ev, "dotnet-ids.token_embeddings")["metrics"]["cases"] == 2
    assert check(ev, "tokenizer.decoder_frame_sha256")["metrics"]["videos"] == 2
    assert check(ev, "tokenizer.image_rgb_sha256")["metrics"]["images"] == 1
    assert check(ev, "export.not_applicable")["status"] == "not-applicable"


def test_an_excluded_row_present_in_the_results_is_evaluated(repo):
    repo.stage1(include_excluded=True)
    repo.stage3(include_excluded=True)
    ev = repo.run(require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK
    assert ev.report["evaluated_although_excluded"] == ["vid-array-no-metadata"]
    assert check(ev, "dotnet-ids.soft_tokens")["metrics"]["cases"] == 3
    assert check(ev, "tokenizer.frame_indices")["metrics"]["videos"] == 3


def test_an_excluded_row_in_one_stage_is_required_in_the_other(repo):
    repo.stage1(include_excluded=True)
    repo.stage3()
    ev = repo.run()
    assert "missing cases ['vid-array-no-metadata']" in check(ev, "dotnet-ids.embedding")["detail"]


def test_without_reference_tensors_g2_cannot_pass(repo):
    repo.stage1()
    repo.stage3()
    ev = repo.run(tensors=False, require_gates=("G1", "G2"))
    assert check(ev, "dotnet-ids.soft_tokens")["status"] == "not-run"
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "not-run"
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED


@pytest.mark.parametrize(
    ("key", "value", "check_id"),
    [
        ("video0_frame_indices", [0, 29], "tokenizer.frame_indices"),
        ("video0_frames_sha256", "x", "tokenizer.frames_sha256"),
        ("video0_resized_sha256", "x", "tokenizer.resized_sha256"),
        ("video0_pixel_values_sha256", "x", "tokenizer.pixel_values_sha256"),
        ("video0_positions_sha256", "x", "tokenizer.positions_sha256"),
        ("video0_frame_sha256", ["x", "y"], "tokenizer.decoder_frame_sha256"),
        ("image0_rgb_sha256", "x", "tokenizer.image_rgb_sha256"),
    ],
)
def test_a_stage_1_mismatch_fails(repo, key, value, check_id):
    def edit(row_id, outs):
        if row_id == "mm-b":
            outs[key] = value

    repo.stage1(edit)
    repo.stage3()
    ev = repo.run()
    assert check(ev, check_id)["status"] == "failed"
    assert "mm-b" in check(ev, check_id)["detail"]
    assert ev.report["first_divergence"] == "tokenizer"
    assert ev.exit_code == evaluate.EXIT_FAILED


def test_rendered_ids_must_be_exact_and_complete(repo):
    def edit(row_id, outs):
        if row_id == "vid-a":
            outs["input_ids"] = outs["input_ids"][:-1]

    repo.stage1(edit, drop="q-c")
    repo.stage3()
    detail = check(repo.run(), "tokenizer.token_ids")["detail"]
    assert "missing cases ['q-c']" in detail and "vid-a" in detail


def test_another_ffmpeg_major_is_report_only_and_g1_cannot_pass(repo):
    repo.stage1(ffmpeg_major=9)
    repo.stage3()
    ev = repo.run(require_gates=("G1", "G2"))
    c = check(ev, "tokenizer.decoder_frame_sha256")
    assert c["status"] == "not-run" and "not the gated major 8" in c["detail"]
    assert c["metrics"]["ffmpeg_major"] == 9
    assert ev.report["gates"]["G1"]["status"] == "not-run"
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED


def test_a_missing_ffmpeg_major_is_not_run(repo):
    repo.stage1(ffmpeg_major=None)
    repo.stage3()
    c = check(repo.run(), "tokenizer.decoder_frame_sha256")
    assert c["status"] == "not-run" and "ffmpeg_major" in c["detail"]


def test_embedding_soft_token_and_batch_bounds(repo):
    repo.stage1()
    repo.stage3(noise=2e-5, soft_noise=0.1, batch_noise=5e-5)
    ev = repo.run()
    assert check(ev, "dotnet-ids.embedding")["status"] == "failed"
    assert check(ev, "dotnet-ids.soft_tokens")["status"] == "failed"  # 0.1 / 16 > 1e-3
    assert check(ev, "dotnet-ids.batch.mixed")["status"] == "failed"
    assert ev.report["first_divergence"] == "dotnet-ids"


def test_a_missing_required_row_fails(repo):
    repo.stage1()
    repo.stage3(drop="mm-b")
    assert "missing cases ['mm-b']" in check(repo.run(), "dotnet-ids.embedding")["detail"]


def test_unknown_tolerance_keys_are_reported(repo):
    tol = copy.deepcopy(TOLERANCES)
    tol["discrete"]["timestamps"] = "exact"
    (repo.protected / "tolerances.yaml").write_text(yaml.safe_dump(tol), encoding="utf-8")
    git(repo.root, "commit", "-qam", "extra key")
    repo.stage1()
    repo.stage3()
    ev = repo.run(require_gates=("G1",))
    c = check(ev, "tokenizer.unknown_tolerance_keys")
    assert "discrete.timestamps" in c["detail"]
    assert ev.report["gates"]["G1"]["status"] == "not-run"


def test_reference_tensors_are_checked_against_the_protected_digests(repo, tmp_path):
    other = tmp_path / "other-tensors"
    other.mkdir()
    for f in repo.tensors.iterdir():
        (other / f.name).write_bytes(f.read_bytes())
    data, _ = npy([9.0] * (FRAMES * SOFT * WIDTH), (FRAMES * SOFT, WIDTH))
    with zipfile.ZipFile(other / "vid-a.npz", "w") as z:
        z.writestr("video0_soft_tokens.npy", data)
        z.writestr("token_embeddings.npy", npy(tokens(ROWS[0]), (len(VIDEO_IDS), WIDTH))[0])
    repo.stage1()
    repo.stage3()
    with pytest.raises(evaluate.EvaluationError, match="tensors.json"):
        evaluate.evaluate(
            repo.manifest, repo.results, repo_root=repo.root, protected_base="HEAD", reference_tensors=other
        )


def test_results_of_another_task_are_rejected(repo):
    repo.stage1()
    raw = json.loads((repo.results / "tokenizer.results.json").read_text())
    raw["task"] = "image-embedding"
    (repo.results / "tokenizer.results.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(evaluate.EvaluationError, match="video-embedding"):
        repo.run()


def test_cli_writes_video_reports(repo, tmp_path):
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
    assert "(video-embedding)" in md and "dotnet-ids.soft_tokens" in md and "per video" in md
    assert "vid-array-no-metadata" in md
    assert json.loads((out / "report.json").read_text())["gates"]["G2"]["status"] == "passed"


def test_expected_keys_match_the_binding_tolerances_layout():
    from model_pipeline.evaluate import expected_tolerance_keys, flatten

    keys = set(expected_tolerance_keys("video-embedding"))
    assert keys == set(flatten(TOLERANCES))
