"""Matched-backend benchmark runner for EmbeddingGemma 2 on Python ONNX Runtime (G5, plan section 11).

The Python side of `ai-dotnet/benchmarks/Starlings.Benchmarks.EmbeddingGemma2Harness`: same graphs, same workload
file (`embeddinggemma2-v1.json`), same fixture rows, same thread settings, same warm/cold protocol and raw format
(`matched_core.py`). Budgets: `embeddinggemma-2/onnx-fp32-{cpu,cuda}` (text), `image-fp32-{cpu,cuda}` and
`video-fp32-{cpu,cuda}` in `model-pipeline/protected/performance-budgets.yaml`; check with `model-pipeline bench g5`.

Text path (`--paths text`; graph `onnx/embeddinggemma2-text-fp32.onnx`), stages as the C# harness:

- `tokenize`: the row's prompt text prepended, Hugging Face `tokenizers` on the same `tokenizer.json` (`<bos>` and
  `<eos>` added, no truncation), one text at a time;
- `encode`: the C# encoder's chunk plan (at most 16 rows and 8,192 padded cells per run), right padding with id 0,
  `InferenceSession.run` to `token_embeddings`, mean pooling over the mask in float64 and L2 normalization.

Image path (`--paths image`; graphs `embeddinggemma2-vision-fp32.onnx` and `embeddinggemma2-text-mm-fp32.onnx`):

- `decode`: Pillow from the file bytes in memory, converted to RGB (the C# harness: SkiaSharp adapter);
- `embed`: the pinned reference preprocessing (transformers 5.19.0 `Gemma4ImageProcessor`, torchvision backend,
  restated as in `reference/embeddinggemma2/run_embeddinggemma2_image.py`: uint8 bicubic antialiased resize to
  the aspect-preserving 48-multiple size within 2,520 patches, x 1/255, 16x16 patches), rendering and tokenizing
  (`<|image>` + n x `<|image|>` + `<image|>` per image, as `EmbeddingGemma2MultimodalInput.Render`), the vision
  graph one image per call on its unpadded patches (the C# defaults: `MaxVisionBatchSize` 1,
  `PadImagesToMaxPatches` false), the soft tokens scattered at the `<|image|>` positions of the multimodal text
  graph (same chunk plan as text), pooling and normalization. Its parts are also recorded as `embed_preprocess`,
  `embed_vision` and `embed_text` (inside `embed`; not added to it);
- `preprocess-only` workload: `decode` and `preprocess`, checked by the pixel-values SHA-256 against the fixture.

Video path (`--paths video`; the same two graphs), on decoded frames: the reference's sampled RGB frames
(`run_embeddinggemma2_video.py --tensors`: `<row id>.npz`, `video<k>_frames_uint8`, checked against the protected
`video-fp32-cpu/tensors.json`), read at load from `$STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS`, so no decoder runs on
either side. `embed` restates the C# video path: per frame the video processor at 140 soft tokens (the 1,260-patch
size rule, uint8 bicubic antialiased resize, x 1/255, 16x16 patches; `eg2_video.preprocess` without the padding),
rendering (`<|image>` + n x `<|video|>` + `<image|>` per frame at each `<|video|>` placeholder, else after the text),
the vision graph on the chunk's frames in calls of at most 2 frames padded to the call's largest frame
(`MaxVideoFramesPerVisionCall` 2, `MaxVisionBatchPatches` 2,520, `PadImagesToMaxPatches` false), the soft tokens at
the `<|video|>` positions of the multimodal text graph, pooling; parts recorded as `embed_preprocess`,
`embed_vision`, `embed_text`. `video-preprocess-only`: `preprocess` alone, checked per video by the pixel-values
SHA-256.

Quality on every series: fixture rows against the protected embeddings (`float.embedding_max_abs` of the profile's
tolerances), synthetic text rows against the same text embedded alone at load
(`float.padded_batch_vs_single_embedding_max_abs`), preprocessing by exact digest.

Usage:
    python bench_eg2_matched.py warm --snapshot <dir> --provider cpu --threads 6 --paths text --out raw.jsonl
    STARLING_EMBEDDINGGEMMA2_IMAGES=<images dir> python bench_eg2_matched.py warm --snapshot <dir> --provider cpu \
        --threads 6 --paths image --out raw.jsonl
    python bench_eg2_matched.py cold --snapshot <dir> --provider cpu --threads 6 --cold-path text --out raw.jsonl
"""

from __future__ import annotations

import time

_T_MAIN_NS = time.monotonic_ns()  # before the heavy imports, for cold-start stages

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_core as core  # noqa: E402

HARNESS = "bench_eg2_matched.py"
PROTECTED = core.PROTECTED / "embeddinggemma-2"
TEXT_FIXTURE = PROTECTED / "onnx-fp32-cpu" / "embeddings.jsonl"
IMAGE_FIXTURES = PROTECTED / "image-fp32-cpu"
TEXT_GRAPH = "embeddinggemma2-text-fp32.onnx"
VISION_GRAPH = "embeddinggemma2-vision-fp32.onnx"
MM_GRAPH = "embeddinggemma2-text-mm-fp32.onnx"
IMAGES_ENV = "STARLING_EMBEDDINGGEMMA2_IMAGES"
VIDEO_FIXTURES = PROTECTED / "video-fp32-cpu"
VIDEO_TENSORS_ENV = "STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS"
IMAGE_TOKEN = 258880
VIDEO_TOKEN = 258884
VIDEO_SOFT_TOKENS, MAX_VIDEO_FRAMES_PER_CALL, MAX_VISION_BATCH_PATCHES = 140, 2, 2520  # MultimodalOptions defaults
MAX_BATCH_SIZE, MAX_BATCH_TOKENS = 16, 8192  # EmbeddingGemma2EncoderOptions / MultimodalOptions defaults
MAX_SOFT_TOKENS, PATCH, POOL = 280, 16, 3
RESCALE = 0.00392156862745098  # processor_config rescale_factor
SIXTEEN_BIT = "mode-gray16"  # rejected by the C# contract
EXIF = "jpeg-exif-orientation"  # excluded from the C# preliminary sweep


def plan_chunks(lengths: list[int], max_rows: int = MAX_BATCH_SIZE, max_cells: int = MAX_BATCH_TOKENS) -> list[int]:
    """EmbeddingGemma2Encoder.PlanChunks: a chunk grows while it has fewer than max_rows rows and rows x longest row
    stays within max_cells; every chunk has at least one row."""
    chunks, count, width = [], 0, 0
    for length in lengths:
        new_width = max(width, length)
        if count > 0 and (count >= max_rows or (count + 1) * new_width > max_cells):
            chunks.append(count)
            count, new_width = 0, length
        count += 1
        width = new_width
    if count:
        chunks.append(count)
    return chunks


def target_size(width: int, height: int, max_soft_tokens: int = MAX_SOFT_TOKENS) -> tuple[int, int]:
    """(height, width) of transformers' get_aspect_ratio_preserving_size (imported from transformers at load)."""
    from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

    return get_aspect_ratio_preserving_size(height, width, PATCH, max_soft_tokens * POOL * POOL, POOL)


def render(prompt_text: str, text: str | None, soft_counts: list[int]) -> str:
    """EmbeddingGemma2MultimodalInput.Render over the fixture row's parts (BuildInput: `<|image|>` in the text marks
    the image positions; without one, the images follow the text)."""
    blocks = ["<|image>" + "<|image|>" * n + "<image|>" for n in soft_counts]
    if text is None:
        return prompt_text + "".join(blocks)
    pieces = text.split("<|image|>")
    if len(pieces) == 1:
        return prompt_text + text + "".join(blocks)
    if len(pieces) - 1 != len(blocks):
        raise SystemExit(f"{len(blocks)} images for {len(pieces) - 1} placeholders")
    out = [prompt_text]
    for i, piece in enumerate(pieces):
        out.append(piece)
        if i < len(blocks):
            out.append(blocks[i])
    return "".join(out)


def render_video(prompt_text: str, text: str | None, frame_counts: list[tuple[int, int]]) -> str:
    """EmbeddingGemma2MultimodalInput.Render for text and videos: each video is, per frame, `<|image>` + n x
    `<|video|>` + `<image|>` (frame_counts: (frames, n) per video), at its `<|video|>` placeholder, else after the
    text."""
    blocks = [("<|image>" + "<|video|>" * n + "<image|>") * frames for frames, n in frame_counts]
    if text is None:
        return prompt_text + "".join(blocks)
    pieces = text.split("<|video|>")
    if len(pieces) == 1:
        return prompt_text + text + "".join(blocks)
    if len(pieces) - 1 != len(blocks):
        raise SystemExit(f"{len(blocks)} videos for {len(pieces) - 1} placeholders")
    out = [prompt_text]
    for i, piece in enumerate(pieces):
        out.append(piece)
        if i < len(blocks):
            out.append(blocks[i])
    return "".join(out)


def compare(np, actual, reference) -> dict:
    act = np.asarray(actual, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    cos = (act * ref).sum(axis=1) / (np.linalg.norm(act, axis=1) * np.linalg.norm(ref, axis=1))
    return {
        "max_abs": float(np.abs(act - ref).max()),
        "min_cosine": float(cos.min()),
        "finite": bool(np.isfinite(act).all()),
    }


def merge(parts: list[dict]) -> dict:
    if not parts:
        return {"max_abs": 0.0, "min_cosine": 1.0, "finite": True}
    return {
        "max_abs": max(p["max_abs"] for p in parts),
        "min_cosine": min(p["min_cosine"] for p in parts),
        "finite": all(p["finite"] for p in parts),
    }


def fixture_dir(kind: str, provider: str) -> Path:
    return PROTECTED / f"{kind}-fp32-{provider}"


class Eg2Run(core.LoadedRun):
    def __init__(self, args: argparse.Namespace, paths: set[str]):
        t0 = time.perf_counter_ns()
        ort = core.import_ort(args.provider)
        import numpy as np
        import tokenizers

        self.np = np
        self.args = args
        self.paths = paths
        stages = {"runtime_init": core.ms_since(t0)}
        t = time.perf_counter_ns()
        self.spec = json.loads(args.workloads.read_text(encoding="utf-8"))
        self.text_rows = {}
        for row in core.read_jsonl(TEXT_FIXTURE):
            key = row["id"] if row["prompt"] == "none" else f"{row['id']}@{row['prompt']}"
            self.text_rows[key] = row
        text_tol = fixture_dir("onnx", args.provider) / "tolerances.yaml"
        self.text_tolerance = core.read_tolerance(text_tol, "float", "embedding_max_abs")
        self.self_tolerance = core.read_tolerance(text_tol, "float", "padded_batch_vs_single_embedding_max_abs")
        self.image_rows: dict[str, dict] = {}
        self.image_bytes: dict[str, bytes] = {}
        self.video_rows: dict[str, dict] = {}
        self.video_frames: dict[str, list] = {}
        if "video" in paths:
            video_tol = fixture_dir("video", args.provider) / "tolerances.yaml"
            self.video_tolerance = core.read_tolerance(video_tol, "float", "embedding_max_abs")
            self.video_rows = {r["id"]: r for r in core.read_jsonl(VIDEO_FIXTURES / "video-embeddings.jsonl")}
            self.video_digests = json.loads((VIDEO_FIXTURES / "tensors.json").read_text())
        if "image" in paths:
            image_tol = fixture_dir("image", args.provider) / "tolerances.yaml"
            self.image_tolerance = core.read_tolerance(image_tol, "float", "embedding_max_abs")
            self.image_rows = {r["id"]: r for r in core.read_jsonl(IMAGE_FIXTURES / "image-embeddings.jsonl")}
            self.manifest = {m["id"]: m for m in json.loads((IMAGE_FIXTURES / "images.json").read_text())["images"]}
        stages["fixture_read"] = core.ms_since(t)
        t = time.perf_counter_ns()
        self.tokenizer = tokenizers.Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        stages["tokenizer_load"] = core.ms_since(t)
        onnx = args.snapshot / "onnx"
        self.text_session = self.vision = self.mm = None
        if "text" in paths:
            t = time.perf_counter_ns()
            self.text_session = core.ort_session(ort, onnx / TEXT_GRAPH, args)
            stages["session_create"] = core.ms_since(t)
        if paths & {"image", "video"}:
            t = time.perf_counter_ns()
            import torch  # (the reference preprocessing)
            from torchvision.transforms.v2 import functional as tvF

            self.tvF = tvF
            sys.path.insert(0, str(core.REPO / "model-pipeline" / "reference" / "embeddinggemma2"))
            from eg2_vision import patchify

            self.patchify = patchify
            self.torch = torch
            target_size(1, 1)  # imports transformers' size rule
            if "video" in paths:
                from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
                    get_aspect_ratio_preserving_size,
                )

                self.video_size = get_aspect_ratio_preserving_size
            stages["preprocess_init"] = core.ms_since(t)
            t = time.perf_counter_ns()
            self.vision = core.ort_session(ort, onnx / VISION_GRAPH, args)
            self.mm = core.ort_session(ort, onnx / MM_GRAPH, args)
            stages["multimodal_create"] = core.ms_since(t)
        self.self_reference: dict[str, object] = {}
        self.load_stages = stages
        session = self.text_session or self.vision
        self.session_providers = session.get_providers()
        self.settings = core.session_settings(args, {k: dict(v) for k, v in session.get_provider_options().items()})
        self.settings.update(
            {
                "pad_id": 0,
                "dimension": 768,
                "paths": sorted(paths),
                "max_batch_size": MAX_BATCH_SIZE,
                "max_batch_tokens": MAX_BATCH_TOKENS,
                "max_image_soft_tokens": MAX_SOFT_TOKENS,
                "max_vision_batch_size": 1,
                "pad_images_to_max_patches": False,
                "pooling": "float64 mean over the mask, L2 normalization",
                "image_preprocessing": "transformers 5.19.0 Gemma4ImageProcessor (torchvision backend), restated",
            }
        )

    # ------------------------------------------------------------------------------------------ text

    def pool(self, tokens, mask):
        np = self.np
        m = mask[:, :, None].astype(np.float64)
        mean = (tokens.astype(np.float64) * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        out = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
        return out.astype(np.float32)

    def pad(self, rows: list[list[int]]):
        np = self.np
        width = max(len(r) for r in rows)
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
            mask[i, : len(r)] = 1
        return ids, mask

    def embed_text(self, request: list[dict]):
        t = time.perf_counter_ns()
        rows = [self.tokenizer.encode(r["prompt_text"] + r["text"]).ids for r in request]
        tokenize = core.ms_since(t)
        t = time.perf_counter_ns()
        out, start = [], 0
        for count in plan_chunks([len(r) for r in rows]):
            ids, mask = self.pad(rows[start : start + count])
            (tokens,) = self.text_session.run(["token_embeddings"], {"input_ids": ids, "attention_mask": mask})
            out.append(self.pool(tokens, mask))
            start += count
        result = self.np.concatenate(out)
        return (result, rows), {"tokenize": tokenize, "encode": core.ms_since(t)}, {}

    def judge_text(self, requests: list[list[dict]]):
        def judge(outputs):
            fixture, synthetic = [], []
            for r, (emb, rows) in outputs:
                for i, row in enumerate(requests[r]):
                    if row["key"] in self.self_reference:
                        synthetic.append(compare(self.np, emb[i : i + 1], self.self_reference[row["key"]]))
                    else:
                        part = compare(self.np, emb[i : i + 1], [row["embedding"]])
                        part["ids_equal"] = rows[i] == row["input_ids"]
                        fixture.append(part)
            q = merge(fixture)
            q.update({"rows": len(fixture), "tolerance": self.text_tolerance})
            ok = q["max_abs"] <= self.text_tolerance and q["finite"] and all(p["ids_equal"] for p in fixture)
            q["ids_equal"] = all(p["ids_equal"] for p in fixture)
            if synthetic:
                s = merge(synthetic)
                q.update({"synthetic_rows": len(synthetic), "synthetic_vs_single_at_load": s})
                q["synthetic_tolerance"] = self.self_tolerance
                ok &= s["max_abs"] <= self.self_tolerance and s["finite"]
                q["finite"] = q["finite"] and s["finite"]
            q["pass"] = bool(ok)
            return q

        return judge

    def text_row(self, key: str) -> dict:
        if key not in self.text_rows:
            raise SystemExit(f"Workload names unknown text row '{key}'.")
        return {**self.text_rows[key], "key": key}

    def synthetic_row(self, synthetic: dict) -> dict:
        source = self.text_row(synthetic["row"])
        words = synthetic["words"]
        text = " ".join([w for w in source["text"].split(" ") if w][:words])
        row = {**source, "text": text, "key": f"{source['id']}:first-{words}-words", "input_ids": None}
        if row["key"] not in self.self_reference:
            self.self_reference[row["key"]] = self.embed_text([row])[0][0]
        return row

    # ----------------------------------------------------------------------------------------- image

    def read_image(self, image_id: str) -> bytes:
        if image_id not in self.image_bytes:
            directory = os.environ.get(IMAGES_ENV)
            if not directory:
                raise SystemExit(f"set {IMAGES_ENV} to the make_eg2_images.py output (images.json and the files)")
            entry = self.manifest[image_id]
            data = (Path(directory) / entry["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise SystemExit(f"{entry['file']}: not the fixture image (SHA-256 differs from images.json)")
            self.image_bytes[image_id] = data
        return self.image_bytes[image_id]

    def decode(self, image_id: str):
        from PIL import Image

        image = Image.open(io.BytesIO(self.image_bytes[image_id]))
        image.load()
        return image if image.mode == "RGB" else image.convert("RGB")

    def preprocess(self, rgb):
        """restate_preprocess (run_embeddinggemma2_image.py) without padding: pixel values, positions, soft tokens."""
        tvF = self.tvF
        chw = tvF.pil_to_tensor(rgb)
        h, w = chw.shape[-2:]
        th, tw = target_size(int(w), int(h))
        resized = (
            chw
            if (th, tw) == (h, w)
            else tvF.resize(chw, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        )
        patches, pos, count = self.patchify(resized * RESCALE, 0, PATCH)
        return patches.numpy(), pos.numpy().astype(self.np.int64), count

    def embed_images(self, request: list[dict]):
        np = self.np
        t = time.perf_counter_ns()
        decoded = {}
        for row in request:
            for image in row["images"]:
                if image["image"] not in decoded:
                    decoded[image["image"]] = self.decode(image["image"])
        decode = core.ms_since(t)
        t_embed = time.perf_counter_ns()
        pre = vision = text = 0.0
        prepared = []
        for row in request:
            t = time.perf_counter_ns()
            patches = [self.preprocess(decoded[i["image"]]) for i in row["images"]]
            pre += core.ms_since(t)
            ids = self.tokenizer.encode(render(row["prompt_text"], row["text"], [p[2] for p in patches])).ids
            prepared.append((ids, patches))
        out, start = [], 0
        for count in plan_chunks([len(p[0]) for p in prepared]):
            chunk = prepared[start : start + count]
            t = time.perf_counter_ns()
            soft_rows = []
            for _, patches in chunk:
                softs = []
                for pv, pos, n in patches:
                    soft, mask = self.vision.run(
                        ["soft_tokens", "soft_token_mask"], {"pixel_values": pv[None], "pixel_position_ids": pos[None]}
                    )
                    if not (mask[0, :n].all() and mask[0].sum() == n):
                        raise SystemExit("the vision graph's soft-token mask is not the first n slots")
                    softs.append(soft[0, :n])
                soft_rows.append(softs)
            vision += core.ms_since(t)
            t = time.perf_counter_ns()
            ids, mask = self.pad([p[0] for p in chunk])
            mm_mask = ids == IMAGE_TOKEN
            mm = np.zeros((ids.shape[0], ids.shape[1], 512), dtype=np.float32)
            for b, softs in enumerate(soft_rows):
                if softs:
                    mm[b][mm_mask[b]] = np.concatenate(softs)
            (tokens,) = self.mm.run(
                ["token_embeddings"],
                {"input_ids": ids, "attention_mask": mask, "multimodal_embeds": mm, "multimodal_mask": mm_mask},
            )
            out.append(self.pool(tokens, mask))
            text += core.ms_since(t)
            start += count
        stages = {
            "decode": decode,
            "embed": core.ms_since(t_embed),
            "embed_preprocess": pre,
            "embed_vision": vision,
            "embed_text": text,
        }
        return (np.concatenate(out), [p[0] for p in prepared]), stages, {}

    def preprocess_only(self, request: list[dict]):
        t = time.perf_counter_ns()
        images = [self.decode(r["images"][0]["image"]) for r in request]
        decode = core.ms_since(t)
        t = time.perf_counter_ns()
        patches = [self.preprocess(i) for i in images]
        preprocess = core.ms_since(t)
        digests = [
            hashlib.sha256(self.np.ascontiguousarray(p[0], dtype=self.np.float32).tobytes()).hexdigest()
            for p in patches
        ]
        return digests, {"decode": decode, "preprocess": preprocess}, {}

    def judge_images(self, requests: list[list[dict]]):
        def judge(outputs):
            parts, ids_equal = [], True
            for r, (emb, ids) in outputs:
                parts.append(compare(self.np, emb, [row["embedding"] for row in requests[r]]))
                ids_equal &= all(i == row["input_ids"] for i, row in zip(ids, requests[r], strict=True))
            q = merge(parts)
            q.update({"rows": sum(len(requests[r]) for r, _ in outputs), "tolerance": self.image_tolerance})
            q["ids_equal"] = ids_equal
            q["pass"] = bool(q["max_abs"] <= self.image_tolerance and q["finite"] and ids_equal)
            return q

        return judge

    @staticmethod
    def judge_preprocess(requests: list[list[dict]]):
        def judge(outputs):
            images = equal = 0
            for r, digests in outputs:
                for i, d in enumerate(digests):
                    images += 1
                    equal += d == requests[r][i]["images"][0]["digests"]["pixel_values"]
            check = "pixel_values SHA-256 equal to the fixture digest (exact)"
            return {"check": check, "images": images, "equal": equal, "pass": equal == images}

        return judge

    def image_row(self, row_id: str) -> dict:
        if row_id not in self.image_rows:
            raise SystemExit(f"Workload names unknown image row '{row_id}'.")
        row = self.image_rows[row_id]
        for image in row["images"]:
            self.read_image(image["image"])
        return row

    # ----------------------------------------------------------------------------------------- video

    def video_row(self, row_id: str) -> dict:
        """A row of text and videos, its sampled frames read once from the reference tensors (outside any timing)."""
        if row_id not in self.video_rows:
            raise SystemExit(f"Workload names unknown video row '{row_id}'.")
        row = self.video_rows[row_id]
        if row["image"] is not None:
            raise SystemExit(f"video row '{row_id}' has a still image; the video workloads take text and videos only")
        if row_id not in self.video_frames:
            directory = os.environ.get(VIDEO_TENSORS_ENV)
            path = Path(directory or "") / f"{row_id}.npz"
            if not directory or not path.exists():
                raise SystemExit(
                    f"set {VIDEO_TENSORS_ENV} to the run_embeddinggemma2_video.py --tensors output ({row_id}.npz)"
                )
            frames = []
            with self.np.load(path) as z:
                for k, video in enumerate(row["videos"]):
                    name = f"video{k}_frames_uint8"
                    array = self.np.ascontiguousarray(z[name])
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    pinned = self.video_digests[row_id][name]["sha256"]
                    if digest != pinned or digest != video["digests"]["frames_uint8"]:
                        raise SystemExit(f"{path}/{name}: not the reference frames (SHA-256 differs from tensors.json)")
                    frames.append(array)
            self.video_frames[row_id] = frames
        return row

    def preprocess_video(self, frames_thwc):
        """The video processor at 140 soft tokens for one video, unpadded: patches [T, P, 768] and positions [T, P, 2]
        (eg2_video.preprocess without padding to 1,260 rows)."""
        torch, tvF = self.torch, self.tvF
        frames = torch.from_numpy(frames_thwc).permute(0, 3, 1, 2).contiguous()
        t, _, h, w = frames.shape
        th, tw = self.video_size(h, w, PATCH, VIDEO_SOFT_TOKENS * POOL * POOL, POOL)
        resized = (
            frames
            if (th, tw) == (h, w)
            else tvF.resize(frames, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        )
        x = resized * RESCALE
        ph, pw = th // PATCH, tw // PATCH
        patches = x.reshape(t, 3, ph, PATCH, pw, PATCH).permute(0, 2, 4, 3, 5, 1).reshape(t, ph * pw, -1)
        gx, gy = torch.meshgrid(torch.arange(pw), torch.arange(ph), indexing="xy")
        pos = torch.stack([gx, gy], dim=-1).reshape(ph * pw, 2)
        return patches.numpy(), pos.numpy().astype(self.np.int64), ph * pw // (POOL * POOL)

    def run_vision_frames(self, frames: list):
        """Frames (pixel values [P, 768], positions [P, 2], soft-token count) through the vision graph in the C# call
        plan (PlanVisionCalls: at most 2 frames and 2,520 patch rows per call, padded to the call's largest frame);
        returns each frame's soft tokens [n, 512]."""
        np = self.np
        out = []
        start = 0
        for count in plan_chunks([f[0].shape[0] for f in frames], MAX_VIDEO_FRAMES_PER_CALL, MAX_VISION_BATCH_PATCHES):
            call = frames[start : start + count]
            rows = max(f[0].shape[0] for f in call)
            pv = np.zeros((count, rows, PATCH * PATCH * 3), dtype=np.float32)
            pos = np.full((count, rows, 2), -1, dtype=np.int64)
            for b, (p, q, _) in enumerate(call):
                pv[b, : p.shape[0]] = p
                pos[b, : q.shape[0]] = q
            feeds = {"pixel_values": pv, "pixel_position_ids": pos}
            soft, mask = self.vision.run(["soft_tokens", "soft_token_mask"], feeds)
            for b, (_, _, n) in enumerate(call):
                if not (mask[b, :n].all() and mask[b].sum() == n):
                    raise SystemExit("the vision graph's soft-token mask is not the first n slots")
                out.append(soft[b, :n])
            start += count
        return out

    def embed_videos(self, request: list[dict]):
        np = self.np
        t_embed = time.perf_counter_ns()
        pre = vision = text = 0.0
        prepared = []
        for row in request:
            t = time.perf_counter_ns()
            videos = [self.preprocess_video(f) for f in self.video_frames[row["id"]]]
            pre += core.ms_since(t)
            rendered = render_video(row["prompt_text"], row["text"], [(v[0].shape[0], v[2]) for v in videos])
            prepared.append((self.tokenizer.encode(rendered).ids, videos))
        out, start = [], 0
        for count in plan_chunks([len(p[0]) for p in prepared]):
            chunk = prepared[start : start + count]
            t = time.perf_counter_ns()
            frames = [(pv[f], pos, n) for _, videos in chunk for pv, pos, n in videos for f in range(pv.shape[0])]
            soft = self.run_vision_frames(frames) if frames else []
            vision += core.ms_since(t)
            t = time.perf_counter_ns()
            ids, mask = self.pad([p[0] for p in chunk])
            mm_mask = ids == VIDEO_TOKEN
            mm = np.zeros((ids.shape[0], ids.shape[1], 512), dtype=np.float32)
            k = 0
            for b, (_, videos) in enumerate(chunk):
                n_frames = sum(v[0].shape[0] for v in videos)
                if n_frames:
                    mm[b][mm_mask[b]] = np.concatenate(soft[k : k + n_frames])
                k += n_frames
            (tokens,) = self.mm.run(
                ["token_embeddings"],
                {"input_ids": ids, "attention_mask": mask, "multimodal_embeds": mm, "multimodal_mask": mm_mask},
            )
            out.append(self.pool(tokens, mask))
            text += core.ms_since(t)
            start += count
        stages = {"embed": core.ms_since(t_embed), "embed_preprocess": pre, "embed_vision": vision, "embed_text": text}
        return (np.concatenate(out), [p[0] for p in prepared]), stages, {}

    def preprocess_videos(self, request: list[dict]):
        np = self.np
        t = time.perf_counter_ns()
        videos = [self.preprocess_video(f) for row in request for f in self.video_frames[row["id"]]]
        preprocess = core.ms_since(t)
        digests = [hashlib.sha256(np.ascontiguousarray(v[0], dtype=np.float32).tobytes()).hexdigest() for v in videos]
        return digests, {"preprocess": preprocess}, {}

    def judge_videos(self, requests: list[list[dict]]):
        def judge(outputs):
            parts, ids_equal = [], True
            for r, (emb, ids) in outputs:
                parts.append(compare(self.np, emb, [row["embedding"] for row in requests[r]]))
                ids_equal &= all(i == row["input_ids"] for i, row in zip(ids, requests[r], strict=True))
            q = merge(parts)
            q.update({"rows": sum(len(requests[r]) for r, _ in outputs), "tolerance": self.video_tolerance})
            q["ids_equal"] = ids_equal
            q["pass"] = bool(q["max_abs"] <= self.video_tolerance and q["finite"] and ids_equal)
            return q

        return judge

    @staticmethod
    def judge_video_preprocess(requests: list[list[dict]]):
        def judge(outputs):
            videos = equal = 0
            for r, digests in outputs:
                expected = [v["digests"]["pixel_values"] for row in requests[r] for v in row["videos"]]
                for d, e in zip(digests, expected, strict=True):
                    videos += 1
                    equal += d == e
            check = "per-video pixel_values SHA-256 (every sampled frame, 140 soft tokens) equal to the fixture digest"
            return {"check": check + " (exact)", "videos": videos, "equal": equal, "pass": equal == videos}

        return judge

    # ---------------------------------------------------------------------------------------- series

    def series(self, name, path, input_, batch, concurrency, samples, warmup, requests, preprocess_only=False):
        if path == "text":
            call, judge, stages = self.embed_text, self.judge_text(requests), ["tokenize", "encode"]
        elif path == "video" and preprocess_only:
            call, judge, stages = self.preprocess_videos, self.judge_video_preprocess(requests), ["preprocess"]
        elif path == "video":
            call, judge = self.embed_videos, self.judge_videos(requests)
            stages = ["embed", "embed_preprocess", "embed_vision", "embed_text"]
        elif preprocess_only:
            call, judge, stages = self.preprocess_only, self.judge_preprocess(requests), ["decode", "preprocess"]
        else:
            call, judge = self.embed_images, self.judge_images(requests)
            stages = ["decode", "embed", "embed_preprocess", "embed_vision", "embed_text"]
        return core.Series(
            workload=name,
            path=path,
            input=input_,
            batch=batch,
            concurrency=concurrency,
            samples=samples,
            warmup_calls=max(warmup, len(requests)),
            requests=len(requests),
            stages=stages,
            call=lambda r: call(requests[r]),
            judge=judge,
        )

    def requests_of(self, w: dict) -> list[list[dict]]:
        if w["path"] == "video":
            requests = [[self.video_row(i) for i in req] for req in w["requests"]]
            if any(len({r["prompt"] for r in req}) != 1 for req in requests):
                raise SystemExit(f"{w['name']}: a batch is one call with one prompt; its rows have different prompts")
            return requests
        if w["path"] == "image":
            return [[self.image_row(i) for i in req] for req in w["requests"]]
        if "synthetic" in w:
            return [[self.synthetic_row(w["synthetic"])]]
        return [[self.text_row(k) for k in req] for req in w["requests"]]

    def warm_series(self):
        for w in self.spec["workloads"]:
            if w["path"] not in self.paths:
                continue
            requests = self.requests_of(w)
            for c in w.get("concurrency", [1]):
                yield self.series(
                    w["name"],
                    w["path"],
                    w["input"],
                    w.get("batch", 1),
                    c,
                    w["samples"],
                    w.get("warmup_calls", self.spec["warmup_calls"]),
                    requests,
                    bool(w.get("preprocess_only")),
                )

    def cold_series(self) -> core.Series:
        path = self.args.cold_path
        keys = self.spec["cold"][path]
        if path == "text":
            return self.series("cold-start", "text", "short", len(keys), 1, 1, 0, [[self.text_row(k) for k in keys]])
        if path == "video":
            return self.series("cold-start", "video", "video", len(keys), 1, 1, 0, [[self.video_row(k) for k in keys]])
        return self.series("cold-start", "image", "photo", len(keys), 1, 1, 0, [[self.image_row(k) for k in keys]])

    def preliminary_quality(self) -> dict:
        everything = self.args.preliminary == "all"
        used = {k for w in self.spec["workloads"] for req in w.get("requests", []) for k in req}
        result: dict = {"scope": "every fixture row" if everything else "the rows the workloads use"}
        references, passed, cases = [], True, 0
        if "text" in self.paths:
            rows = [self.text_row(k) for k in self.text_rows if everything or k in used]
            singles = [[r] for r in rows]
            q1 = self.judge_text(singles)([(i, self.embed_text(req)[0]) for i, req in enumerate(singles)])
            short = [r for r in rows if len(r["input_ids"]) < 512]
            batches = [short[i : i + 8] for i in range(0, len(short), 8)]
            q8 = self.judge_text(batches)([(i, self.embed_text(req)[0]) for i, req in enumerate(batches)])
            result.update({"text_single": q1, "text_batch8": q8})
            references.append(core.repo_relative(TEXT_FIXTURE))
            passed &= q1["pass"] and q8["pass"]
            cases += len(rows)
        if "image" in self.paths:
            rows = [
                r
                for k, r in self.image_rows.items()
                if r["images"]
                and all(i["image"] not in (SIXTEEN_BIT, EXIF) for i in r["images"])
                and (everything or k in used)
            ]
            singles = [[self.image_row(r["id"])] for r in rows]
            q = self.judge_images(singles)([(i, self.embed_images(req)[0]) for i, req in enumerate(singles)])
            q["text_only_rows_not_run"] = sum(1 for r in self.image_rows.values() if not r["images"])
            result["image_single"] = q
            references.append(core.repo_relative(IMAGE_FIXTURES / "image-embeddings.jsonl"))
            passed &= q["pass"]
            cases += len(rows)
        if "video" in self.paths:
            rows = [
                r
                for k, r in self.video_rows.items()
                if r["image"] is None and r["videos"] and (everything or k in used)
            ]
            singles = [[self.video_row(r["id"])] for r in rows]
            q = self.judge_videos(singles)([(i, self.embed_videos(req)[0]) for i, req in enumerate(singles)])
            q["rows_not_run"] = len(self.video_rows) - len(rows)
            result["video_single"] = q
            references.append(core.repo_relative(VIDEO_FIXTURES / "video-embeddings.jsonl"))
            passed &= q["pass"]
            cases += len(rows)
        result.update({"reference": "; ".join(references), "cases": cases, "pass": bool(passed)})
        return result


class Eg2Driver(core.Driver):
    model = "embeddinggemma-2"
    harness = HARNESS
    default_workloads = core.WORKLOADS / "embeddinggemma2-v1.json"
    snapshot_env = "STARLING_EMBEDDINGGEMMA2_SNAPSHOT"

    def add_arguments(self, parser: argparse.ArgumentParser, command: str) -> None:
        parser.add_argument("--paths", default="text", help="text, image and/or video (default text)")
        if command in ("cold", "cold-child"):
            parser.add_argument("--cold-path", choices=["text", "image", "video"], default="text")
        else:
            parser.add_argument("--preliminary", choices=["all", "workloads"], default="all")

    def forwarded(self, args: argparse.Namespace) -> list[str]:
        return ["--paths", args.paths, "--cold-path", args.cold_path]

    @staticmethod
    def paths(args: argparse.Namespace) -> set[str]:
        paths = set(args.paths.split(","))
        if not paths or paths - {"text", "image", "video"}:
            raise SystemExit("--paths takes text, image and/or video")
        return paths

    def profile(self, args: argparse.Namespace) -> str:
        kinds = {"text": "onnx", "image": "image", "video": "video"}
        order = ["text", "image", "video"]
        return "+".join(f"{kinds[p]}-fp32-{args.provider}" for p in sorted(self.paths(args), key=order.index))

    def load(self, args: argparse.Namespace) -> core.LoadedRun:
        if args.snapshot is None:
            raise SystemExit("--snapshot or STARLING_EMBEDDINGGEMMA2_SNAPSHOT is required")
        return Eg2Run(args, self.paths(args))

    def reference(self, args: argparse.Namespace) -> str:
        refs = {
            "text": TEXT_FIXTURE,
            "image": IMAGE_FIXTURES / "image-embeddings.jsonl",
            "video": VIDEO_FIXTURES / "video-embeddings.jsonl",
        }
        return "; ".join(core.repo_relative(refs[p]) for p in sorted(self.paths(args)))

    def artifacts(self, args: argparse.Namespace) -> dict:
        paths = self.paths(args)
        onnx = args.snapshot / "onnx"
        fixtures = [TEXT_FIXTURE, PROTECTED / "onnx-fp32-cpu" / "tolerances.yaml"]
        extra = {}
        if "video" in paths:
            fixtures += [VIDEO_FIXTURES / n for n in ("video-embeddings.jsonl", "tensors.json", "tolerances.yaml")]
        if paths & {"image", "video"}:
            if "image" in paths:
                fixtures += [IMAGE_FIXTURES / n for n in ("image-embeddings.jsonl", "images.json", "tolerances.yaml")]
            extra["vision_model_sha256"] = core.sha256_file(onnx / VISION_GRAPH)
            extra["multimodal_model_sha256"] = core.sha256_file(onnx / MM_GRAPH)
        packages = ("onnxruntime", "onnxruntime-gpu", "numpy", "tokenizers", "pillow", "torch", "torchvision")
        return core.artifacts(
            onnx / TEXT_GRAPH,
            args.snapshot / "tokenizer.json",
            args.workloads,
            core.file_digests(*fixtures),
            core.package_versions((*packages, "transformers")),
            extra,
        )


def main() -> int:
    return core.main(Eg2Driver(), Path(__file__).resolve(), _T_MAIN_NS, __doc__.split("\n", 1)[0])


if __name__ == "__main__":
    raise SystemExit(main())
