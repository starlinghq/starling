"""Staged fidelity evaluator, task `video-embedding` (EmbeddingGemma 2 videos; extends P-007 and the image task).

Stages (same substage names as the other tasks, so gates and reports line up):

    tokenizer    stage 1, decoding, sampling, rendering and preprocessing: rendered ids (placeholder expansion and
                 the owned tokenizer); per video the sampled frame indices, the SHA-256 of the sampled RGB frames
                 (T x H x W x 3) and of each frame (the FFmpeg decoder profile, gated only on the supported FFmpeg
                 major), the resized frames, the pixel values and the positions; the still image's RGB for rows
                 with one; all exact against `video-embeddings.jsonl` and `tensors.json`                   (G1)
    export       not applicable: the video tolerances bind the delivered C# path; the Python ONNX Runtime
                 parity of the graphs is protected evidence of its own (`parity-cpu.json`)
    dotnet-ids   stage 3, the C# implementation from decoded frames: embeddings (max abs, min cosine), soft
                 tokens per video and token embeddings per media row (row-relative: max |d| / max |ref|), batches
                 against single rows, NaN/Inf, provider options and the assets it loaded                   (G2)
    dotnet-text  not applicable (text-only rows are part of stage 3: they run on the released text graph)

The row `vid-array-no-metadata` (upstream embedding a decoded array without timing metadata) is excluded by contract:
the C# package rejects frames without timing unless `AllowVideosWithoutTiming` is set. It is not required; when a
stage's results contain it (a run with the opt-in), it is evaluated in every stage like any other row.

Soft tokens and token embeddings are compared with the reference tensors, which are too large for Git: the `.npz`
files of `run_embeddinggemma2_video.py --tensors`, each array checked against the SHA-256 in the protected
`tensors.json` before use (`reference_tensors`, CLI `--reference-tensors`). Without them those two checks are
not-run, so G2 cannot pass.

Decoder profiles, one per FFmpeg adapter package (decision D3): tolerance section `decoder_ffmpeg` binds the FFmpeg 8
adapter and `decoder_ffmpeg9` the FFmpeg 9 adapter; each holds `decoded_rgb_vs_torchcodec` and `supported_ffmpeg_major`.
One evaluation holds decoder results for several majors: the tokenizer stage's own frames (`video<k>_frame_sha256`, the
major in `environment.ffmpeg_major`) and, per further adapter, `video<k>_frame_sha256_ffmpeg<major>` with an entry
`{"ffmpeg_major": <major>, "ffmpeg_version": ...}` in `environment.decoders`. Each section present in the protected
tolerances is checked against the results of its major (check `tokenizer.decoder_frame_sha256` for `decoder_ffmpeg`,
`tokenizer.decoder_ffmpeg9_frame_sha256` for `decoder_ffmpeg9`) and is not-run when the results hold no frames of that
major, so a bound nobody checked never passes. The expected digests are the protected `frame_sha256` (torchcodec on
FFmpeg 8); torchcodec 0.17.0 on FFmpeg 9.0.1 gives the same digests for every frame (`torchcodec_eg2_frames.py`). A
section absent from the protected tolerances adds no check.

Tolerance keys (flattened YAML paths) read from the profile's protected tolerances file only: `EXPECTED_KEYS`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_pipeline import evaluate_image as im
from model_pipeline import results
from model_pipeline.evaluate import Check, EvaluationError, _bound, substage_status

TASK = "video-embedding"
SUBSTAGES = im.SUBSTAGES
SUBSTAGE_TITLES = {
    "tokenizer": "Stage 1, decoding, sampling, rendering and preprocessing (indices, frames, ids, pixels, positions)",
    "export": "Stage 2, export (not applicable: Python ORT parity is protected evidence of its own)",
    "dotnet-ids": "Stage 3, .NET from decoded frames (embeddings, soft tokens, token embeddings, batches)",
    "dotnet-text": "Stage 4 (not applicable: text-only rows run in stage 3 on the released text graph)",
}

EXPECTED_KEYS: dict[str, Any] = {
    "profile": "<family>/<profile>, must equal the manifest id",
    "provider_options": "{} on CPU; {use_tf32: 0} on CUDA",
    "discrete.frame_indices": "exact",
    "discrete.token_ids": "exact",
    "discrete.decoded_frames": "exact",
    "discrete.resized_pixels": "exact",
    "discrete.pixel_values": "exact",
    "discrete.video_positions": "exact",
    "float.soft_tokens_row_relative": 1.0e-3,
    "float.token_embeddings_row_relative": 1.0e-3,
    "float.embedding_max_abs": 1.0e-5,
    "float.min_cosine": 0.9999995,
    "float.padded_batch_vs_single_embedding_max_abs": 1.0e-5,
    "float.nan_or_inf": "fail",
    "float.report_only": ["mean_abs", "p99_abs", "per_value_relative"],
    "decoder_ffmpeg.decoded_rgb_vs_torchcodec": "exact",
    "decoder_ffmpeg.supported_ffmpeg_major": 8,
    # The FFmpeg 9 adapter's decoder profile (W4; text proposed for the maintainer to commit). Checked when present.
    "decoder_ffmpeg9.decoded_rgb_vs_torchcodec": "exact",
    "decoder_ffmpeg9.supported_ffmpeg_major": 9,
}
KNOWN_KEYS = set(EXPECTED_KEYS)
# Decoder tolerance sections: (section, check id, required). A required section is checked (not-run when unset); an
# optional one only when the protected tolerances carry it.
DECODER_SECTIONS = (
    ("decoder_ffmpeg", "tokenizer.decoder_frame_sha256", True),
    ("decoder_ffmpeg9", "tokenizer.decoder_ffmpeg9_frame_sha256", False),
)
EXCLUDED_ROWS = {
    "vid-array-no-metadata": "decoded frames without timing metadata (rejected by contract unless opted in)",
}

# (stage-1 output suffix, fixture digest name, tolerance key)
DIGESTS = (
    ("frames_sha256", "frames_uint8", "discrete.decoded_frames"),
    ("resized_sha256", "resized_uint8", "discrete.resized_pixels"),
    ("pixel_values_sha256", "pixel_values", "discrete.pixel_values"),
    ("positions_sha256", "positions", "discrete.video_positions"),
)


@dataclass
class Row:
    id: str
    input_ids: list[int]
    embedding: list[float]
    videos: list[dict[str, Any]]
    image: dict[str, Any] | None

    @property
    def has_media(self) -> bool:
        return bool(self.videos) or self.image is not None


@dataclass
class Fixtures:
    rows: dict[str, Row]
    excluded: dict[str, str]
    tensor_digests: dict[str, dict[str, Any]]
    files: list[Path] = field(default_factory=list)
    optional_included: set[str] = field(default_factory=set)

    @property
    def case_ids(self) -> list[str]:
        return [r for r in self.rows if r not in self.excluded or r in self.optional_included]


def load_fixtures(directory: Path) -> Fixtures:
    emb, vids, tens = (directory / n for n in ("video-embeddings.jsonl", "videos.json", "tensors.json"))
    for p in (emb, vids, tens):
        if not p.is_file():
            raise EvaluationError(f"{p}: missing video fixture file")
    rows: dict[str, Row] = {}
    for line in emb.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["id"]] = Row(r["id"], r["input_ids"], r["embedding"], r.get("videos") or [], r.get("image"))
    excluded = {k: v for k, v in EXCLUDED_ROWS.items() if k in rows}
    return Fixtures(rows, excluded, json.loads(tens.read_text(encoding="utf-8")), [emb, vids, tens])


class VideoEvaluator(im.ImageEvaluator):
    """Checks for a `video-embedding` profile. One instance per evaluation."""

    def __init__(
        self,
        man: dict[str, Any],
        fx: Fixtures,
        tol: dict[str, Any],
        stages: dict[str, results.StageResults],
        reference_tensors: Path | None = None,
    ):
        super().__init__(man, fx, tol, stages, reference_tensors)  # type: ignore[arg-type]
        # An excluded row present in any stage's results is evaluated in every stage.
        fx.optional_included = {c for c in fx.excluded for s in stages.values() if c in s.cases}

    # -- stage 1 ------------------------------------------------------------------------------

    def _per_video(self, check: Check, stage: results.StageResults, suffix: str, expected_of) -> None:
        videos = mismatched = 0
        missing, differ = [], []
        for cid in self.fx.case_ids:
            for k, video in enumerate(self.fx.rows[cid].videos):
                expected = expected_of(video)
                if expected is None:
                    continue
                videos += 1
                got = stage.cases.get(cid, {}).get(f"video{k}_{suffix}")
                if got is None:
                    missing.append(f"{cid}/video{k}")
                elif (list(got) if isinstance(got, list) else got) != expected:
                    mismatched += 1
                    differ.append(f"{cid}/video{k}")
        check.metrics = {**check.metrics, "videos": videos, "cases": videos, "mismatched": mismatched}
        if missing or differ:
            check.status = "failed"
            check.detail = "; ".join(
                x for x in (f"missing {missing}" if missing else None, f"differ {differ}" if differ else None) if x
            )
        else:
            check.status = "passed"
        self._add(check)

    def check_preprocessing(self) -> None:
        stage = self.stages.get("tokenizer")
        names = (
            ["token_ids", "frame_indices"]
            + [suffix for suffix, _, _ in DIGESTS]
            + ["image_rgb_sha256", "decoder_frame_sha256"]
        )
        if stage is None:
            for name in names:
                self._add(Check(f"tokenizer.{name}", "tokenizer", name, detail="no results for stage 'tokenizer'"))
            return

        ids = Check("tokenizer.token_ids", "tokenizer", "Rendered ids (placeholder expansion + tokenizer) exact")
        if self._exact_key(ids, "discrete.token_ids"):
            have = {c for c, o in stage.cases.items() if "input_ids" in o}
            problem = self._coverage(have, stage)
            bad = [
                c for c in self.fx.case_ids if c in have and stage.cases[c]["input_ids"] != self.fx.rows[c].input_ids
            ]
            ids.metrics = {"cases": len(have & set(self.fx.case_ids)), "excluded_by_contract": self.fx.excluded}
            if problem or bad:
                ids.status = "failed"
                ids.detail = "; ".join(x for x in (problem, f"ids differ in {bad}" if bad else None) if x)
            else:
                ids.status = "passed"
            self._add(ids)

        idx = Check("tokenizer.frame_indices", "tokenizer", "Per video sampled frame indices exact")
        if self._exact_key(idx, "discrete.frame_indices"):
            self._per_video(idx, stage, "frame_indices", lambda v: v["sampled_indices"])

        for suffix, digest, key in DIGESTS:
            check = Check(f"tokenizer.{suffix}", "tokenizer", f"Per video {digest} digest exact")
            if self._exact_key(check, key):
                self._per_video(check, stage, suffix, lambda v, d=digest: v["digests"][d])

        still = Check("tokenizer.image_rgb_sha256", "tokenizer", "Still image (a decoded frame) RGB digest exact")
        if self._exact_key(still, "discrete.decoded_frames"):
            missing, differ, n = [], [], 0
            for cid in self.fx.case_ids:
                if self.fx.rows[cid].image is None:
                    continue
                n += 1
                expected = self.fx.tensor_digests.get(cid, {}).get("image0_rgb", {}).get("sha256")
                got = stage.cases.get(cid, {}).get("image0_rgb_sha256")
                if got is None:
                    missing.append(cid)
                elif got != expected:
                    differ.append(cid)
            still.metrics = {"images": n, "mismatched": len(differ)}
            still.status = "failed" if missing or differ else "passed"
            still.detail = "; ".join(
                x for x in (f"missing {missing}" if missing else None, f"differ {differ}" if differ else None) if x
            )
            self._add(still)

        for section, check_id, required in DECODER_SECTIONS:
            if required or any(k.startswith(f"{section}.") for k in self.tol):
                self._decoder(stage, section, check_id)

    @staticmethod
    def _decoder_results(stage: results.StageResults) -> dict[str, tuple[str, dict[str, Any]]]:
        """Decoder results by FFmpeg major (as a string): (case output suffix, environment of that decoder)."""
        found: dict[str, tuple[str, dict[str, Any]]] = {}
        env = stage.environment
        if env.get("ffmpeg_major") is not None:
            found[str(env["ffmpeg_major"])] = ("frame_sha256", env)
        for extra in env.get("decoders") or []:
            major = extra.get("ffmpeg_major") if isinstance(extra, dict) else None
            if major is not None and str(major) not in found:
                found[str(major)] = (f"frame_sha256_ffmpeg{major}", extra)
        return found

    def _decoder(self, stage: results.StageResults, section: str, check_id: str) -> None:
        key = f"{section}.decoded_rgb_vs_torchcodec"
        title = (
            "FFmpeg decoder: each sampled frame's RGB exact"
            if section == "decoder_ffmpeg"
            else f"FFmpeg decoder ({section}): each sampled frame's RGB exact"
        )
        check = Check(check_id, "tokenizer", title)
        if not self._exact_key(check, key):
            return
        want = self.tol.get(f"{section}.supported_ffmpeg_major")
        decoders = self._decoder_results(stage)
        if want is not None and str(want) in decoders:
            suffix, env = decoders[str(want)]
            check.metrics = {"ffmpeg_major": env.get("ffmpeg_major"), "ffmpeg_version": env.get("ffmpeg_version")}
            self._per_video(check, stage, suffix, lambda v: v.get("frame_sha256") or None)
            return
        have = stage.environment.get("ffmpeg_major")
        check.metrics = {"ffmpeg_major": have, "ffmpeg_version": stage.environment.get("ffmpeg_version")}
        if want is None:
            check.detail = f"{section}.supported_ffmpeg_major is not set in the protected tolerances"
            self._add(check)
            return
        if have is None:
            check.detail = (
                "stage 'tokenizer' environment has no ffmpeg_major"
                if not decoders
                else f"no decoder results for FFmpeg major {want} (have {sorted(decoders)})"
            )
            self._add(check)
            return
        self._per_video(check, stage, "frame_sha256", lambda v: v.get("frame_sha256") or None)
        if str(have) != str(want):
            # Report-only: another FFmpeg major is outside the gated decoder profile.
            check.status = "not-run"
            check.detail = (
                f"FFmpeg major {have} is not the gated major {want}; decoded frames reported only "
                f"({check.metrics.get('mismatched', 0)} of {check.metrics.get('videos', 0)} videos differ)"
                + (f"; {check.detail}" if check.detail else "")
            )

    # -- stage 3 ------------------------------------------------------------------------------

    def _tensors(self, stage: results.StageResults, non_finite: list[str]) -> None:
        for name, key in (
            ("soft_tokens", "float.soft_tokens_row_relative"),
            ("token_embeddings", "float.token_embeddings_row_relative"),
        ):
            check = Check(f"dotnet-ids.{name}", "dotnet-ids", f"{name}, row-relative max |d| / max |ref|")
            if name not in stage.tensor_specs:
                check.tolerance_key, check.detail = key, f"stage 'dotnet' has no {name!r} tensor"
                self._add(check)
                continue
            tensor = stage.tensor(name)
            if self.reference_tensors is None:
                check.tolerance_key = key
                check.detail = (
                    "reference tensors not given (--reference-tensors <dir of run_embeddinggemma2_video.py .npz>)"
                )
                self._add(check)
                continue
            worst, worst_id, compared, missing = 0.0, "", 0, []
            for cid in self.fx.case_ids:
                row = self.fx.rows[cid]
                if name == "soft_tokens":
                    pairs = [(f"{cid}/video{k}", f"video{k}_soft_tokens") for k in range(len(row.videos))]
                elif row.has_media:
                    pairs = [(cid, "token_embeddings")]
                else:
                    pairs = []  # text-only rows run on the released text graph (onnx-fp32-cpu evidence)
                for tid, ref in pairs:
                    if tid not in tensor.values:
                        missing.append(tid)
                        continue
                    values = tensor.values[tid]
                    if any(not math.isfinite(v) for v in values):
                        non_finite.append(tid)
                        continue
                    digest = self.fx.tensor_digests[cid][ref]["sha256"]
                    shape, expected = im.read_npz_float32(self.reference_tensors / f"{cid}.npz", ref, digest)
                    if list(tensor.shapes[tid]) != shape:
                        missing.append(f"{tid} (shape {tensor.shapes[tid]} vs {shape})")
                        continue
                    rel = im._row_relative(values, expected)
                    compared += 1
                    if rel > worst:
                        worst, worst_id = rel, tid
            check.metrics = {"max_row_relative": worst, "worst_case": worst_id, "cases": compared}
            if missing:
                check.tolerance_key, check.bound = key, _bound(self.tol, key)
                check.status, check.detail = "failed", f"missing or misshapen: {missing}"
                self._add(check)
            else:
                self._judge(check, key, worst, worst_id)

    def check_not_applicable(self) -> None:
        for substage, why in (
            ("export", "the video tolerances bind the C# path; Python ORT parity: protected parity-cpu.json"),
            ("dotnet-text", "text-only rows are stage 3 rows (released text graph)"),
        ):
            self._add(
                Check(
                    f"{substage}.not_applicable",
                    substage,
                    SUBSTAGE_TITLES[substage],
                    status="not-applicable",
                    detail=why,
                )
            )

    def check_unknown_keys(self) -> None:
        unknown = sorted(k for k in self.tol if k not in KNOWN_KEYS and k != "reference")
        if unknown:
            gate_sub = (
                "tokenizer" if all(k.startswith(("discrete.", "decoder_ffmpeg.")) for k in unknown) else "dotnet-ids"
            )
            self._add(
                Check(
                    f"{gate_sub}.unknown_tolerance_keys",
                    gate_sub,
                    "Tolerance keys this evaluator does not implement",
                    detail=f"not evaluated: {unknown}",
                )
            )


def diagnose(checks: list[Check]) -> tuple[str | None, list[dict[str, str]]]:
    status = {s: substage_status(checks, s) for s in SUBSTAGES}
    first = next((s for s in SUBSTAGES if status[s] == "failed"), None)
    findings = []
    for c in checks:
        if c.status == "invalid":
            findings.append({"stage": c.substage, "taxonomy": "packaging/harness", "finding": f"{c.id}: {c.detail}"})
        elif c.status == "failed":
            taxonomy = {
                "tokenizer": "decoding, sampling, rendering or preprocessing",
                "dotnet-ids": "numerics" if c.tolerance_key and c.tolerance_key.startswith("float.") else "integration",
            }.get(c.substage, "other")
            findings.append({"stage": c.substage, "taxonomy": taxonomy, "finding": f"{c.id}: {c.detail}"})
    return first, findings


def markdown(report: dict[str, Any]) -> str:
    text = im.markdown(report)
    for substage, title in im.SUBSTAGE_TITLES.items():
        text = text.replace(f"| {title} |", f"| {SUBSTAGE_TITLES[substage]} |")
    text = text.replace(
        "Row-relative: per image (soft tokens) or row (token embeddings)",
        "Row-relative: per video (soft tokens) or row (token embeddings)",
    )
    if report.get("excluded_by_contract"):
        text += "\nExcluded by contract (evaluated only when present in the results): " + ", ".join(
            f"`{k}` ({v})" for k, v in report["excluded_by_contract"].items()
        )
        text += "\n"
    return text
