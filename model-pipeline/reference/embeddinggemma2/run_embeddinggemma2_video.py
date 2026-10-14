"""Reference runner for google/embeddinggemma-2, video path (EmbeddingGemma 2 video-path spike). CPU, fp32.

Inputs: a local snapshot pinned by Hub commit SHA and the video set from `make_eg2_videos.py` (`videos.json`).
Upstream is sentence-transformers 6.1.0 over transformers 5.19.0 with torchcodec 0.17.0 decoding through FFmpeg
(the pinned LGPL build of `fetch_eg2_ffmpeg.sh` on LD_LIBRARY_PATH; `eg2_video.pin_torchcodec_ffmpeg`). The flow is
restated in `eg2_video.py`; this runner checks the restatement against the processor bit for bit and stops if any
sampled index, decoded frame, resized pixel, patch or position differs.

Writes into `--out` (small; fixture candidates):

- video-embeddings.jsonl: one row per input (video only; text with a video; two videos; video and image; prompts;
  pre-decoded frame arrays with and without metadata; a torchcodec `VideoDecoder` object; text queries): rendered
  ids, per video the torchcodec metadata, sampled indices, decoded and resized sizes, soft tokens per frame, SHA-256
  of the decoded frames (each and all), the resized frames and every model intermediate, and the fp32 embedding.
- video-frames.json: per video, the SHA-256 of every decoded RGB frame (all frames, not only the sampled ones), for
  decoder comparisons.
- video-reference.json: stability (single video vs batch, eager vs sdpa, restated vision path vs upstream, unpadded
  vs padded, frames through the vision tower one at a time vs together), a query-by-video cosine table, and the
  upstream quirks observed (`.webm` path strings, arrays without metadata).
- video-environment.json: interpreter, packages, FFmpeg libraries, asset and video digests.

Writes into `--tensors` (large; never committed): one `.npz` per row with the sampled decoded frames, the resized
frames, positions, pooled and projected soft tokens, token embeddings and the embedding; `tensors.json` with digests.

Usage (LD_LIBRARY_PATH=<LGPL FFmpeg lib dir>):
    python run_embeddinggemma2_video.py --model-dir <snapshot> --revision <sha> --videos <dir> --out <dir>
        --tensors <dir> [--cases <cases.json>] [--skip-stability]

`--cases` replaces every development row and check that names development media (`MULTIMODAL`, the decoded-array
and decoder-object rows, `QUERIES`, and the video lists of the batch and eager checks) with a JSON file, so the
runner can run on another media set (for example the P-030 release media fetched by `fetch_eg2_release_media.py`).
Every video of `videos.json` still gets its path row. The file:

    {"multimodal": [{"id": str, "text": str | null, "videos": [video id], "image": [video id, frame index] | null,
                     "prompt": "none" | "query" | "document"}],
     "forms": [{"id": str, "video": video id, "form": "array+metadata" | "array" | "decoder"}],
     "queries": [{"id": str, "text": str}],
     "batch": [video id], "eager": [video id]}

Each form appears at most once. The decoded-array forms hold every frame of the video in memory: name a short,
small video. With `--cases`, the `infer_modality` quirk check uses the first video of `videos.json`.
"""

from __future__ import annotations

import eg2_video

eg2_video.pin_torchcodec_ffmpeg()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchcodec._core as tc_core  # noqa: E402
from eg2_video import POOL, VIDEO_TOKEN  # noqa: E402
from eg2_vision import VisionPath  # noqa: E402
from PIL import Image  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sentence_transformers.base.modality import infer_modality  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

from model_pipeline import fixtures  # noqa: E402

REPO = "google/embeddinggemma-2"
PROMPTS = {"none": "", "query": "task: search result | query: ", "document": "title: none | text: "}
IMAGE_TOKEN = 258880
BARS, SHAPES, VP9, TEXT = (
    "h264-bars-640x360-30fps-5s",
    "h264-shapes-1280x720-25fps-10s",
    "vp9-640x360-30fps-6s",
    "h264-ntsc-640x480-29.97fps-12s",
)
# Multimodal rows: (id, text or None, video ids, image source (video id, frame index) or None, prompt name).
MULTIMODAL = [
    ("mm-caption-then-video", "Colour bars with a moving white square.", [BARS], None, "none"),
    ("mm-placeholder", "The clip <|video|> shows moving circles and rectangles.", [SHAPES], None, "none"),
    ("mm-two-videos", "<|video|> and then <|video|>", [BARS, VP9], None, "none"),
    ("mm-image-and-video", "A still <|image|> taken from the clip <|video|>", [SHAPES], (SHAPES, 100), "none"),
    ("mm-document-prompt", "Scrolling words over a gradient.", [TEXT], None, "document"),
    ("vid-query-prompt", None, [BARS], None, "query"),
]
QUERIES = [
    ("q-bars", "colour bars test pattern with a moving white square"),
    ("q-shapes", "colourful circles and rectangles moving over a dark background"),
    ("q-text", "white words scrolling across a colour gradient"),
    ("q-tiny", "a tiny low resolution clip"),
    ("q-cat", "a photo of a cat"),
]
PACKAGES = ["torch", "torchvision", "torchcodec", "transformers", "sentence-transformers", "tokenizers", "numpy"]
ASSETS = ["config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json", "model.safetensors"]


def sha(a) -> str:
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(diff) -> dict:
    a = np.abs(np.asarray(diff, dtype=np.float64)).ravel()
    return {"max_abs": float(a.max()) if a.size else 0.0, "mean_abs": float(a.mean()) if a.size else 0.0}


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def load(model_dir: Path, attn: str | None = None) -> SentenceTransformer:
    model_kwargs = {"dtype": torch.float32}
    if attn:
        model_kwargs["attn_implementation"] = attn
    st = SentenceTransformer(str(model_dir), device="cpu", model_kwargs=model_kwargs)
    st.eval()
    return st


class Capture:
    """Forward hooks: vision encoder, vision tower (pooled), embed_vision (soft tokens), language model."""

    def __init__(self, model: torch.nn.Module):
        self.data: dict[str, list[torch.Tensor]] = {}
        self.handles = [
            self._hook(model.vision_tower.encoder, "encoder"),
            self._hook(model.vision_tower, "pooled"),
            self._hook(model.embed_vision, "soft_tokens"),
            self._hook(model.language_model, "token_embeddings"),
        ]

    def _hook(self, module, name):
        def fn(_m, _i, out):
            t = out if isinstance(out, torch.Tensor) else out.last_hidden_state
            self.data.setdefault(name, []).append(t.detach().clone())

        return module.register_forward_hook(fn)

    def close(self):
        for h in self.handles:
            h.remove()


def render(tok, ids: list[int]) -> str:
    """Decoded sequence with each run of `<|video|>` or `<|image|>` written once with its length."""
    parts, chunk, run, run_id = [], [], 0, None
    names = {VIDEO_TOKEN: "<|video|>", IMAGE_TOKEN: "<|image|>"}
    for i in [*ids, None]:
        if i in names and (run_id in (None, i)):
            if chunk:
                parts.append(tok.decode(chunk, skip_special_tokens=False))
                chunk = []
            run, run_id = run + 1, i
            continue
        if run:
            parts.append(f"{names[run_id]}x{run}")
            run, run_id = 0, None
        if i in names:
            run, run_id = 1, i
            continue
        if i is not None:
            chunk.append(i)
    if chunk:
        parts.append(tok.decode(chunk, skip_special_tokens=False))
    return "".join(parts)


def decode_video(path: Path) -> dict:
    """Upstream decoding (as `read_video_torchcodec`): metadata, sampled indices, sampled frames, all-frame digests."""
    d = VideoDecoder(str(path), seek_mode="exact", num_ffmpeg_threads=0)
    m = d.metadata
    indices = eg2_video.sample_indices(m.num_frames, m.average_fps, m.duration_seconds)
    sampled = d.get_frames_at(indices=indices).data.contiguous()  # uint8 [T, 3, H, W]
    digests = []
    for s in range(0, m.num_frames, 64):
        batch = d.get_frames_in_range(s, min(s + 64, m.num_frames)).data
        digests += [sha(f.permute(1, 2, 0)) for f in batch]
    return {
        "metadata": {
            "num_frames": m.num_frames,
            "average_fps": m.average_fps,
            "duration_seconds": m.duration_seconds,
            "begin_stream_seconds": m.begin_stream_seconds,
            "end_stream_seconds": m.end_stream_seconds,
            "width": m.width,
            "height": m.height,
            "codec": m.codec,
            "rotation": getattr(m, "rotation", None),
        },
        "indices": indices,
        "frames": sampled,
        "all_frame_sha256": digests,
    }


def video_arg(path: Path):
    """Path string where sentence-transformers detects a video; otherwise a `{"video": path}` dict."""
    return str(path) if infer_modality(str(path)) == "video" else {"video": str(path)}


def modality_by_extension(video: Path) -> dict:
    """`infer_modality` on a symlink to an existing video under each common extension."""
    import tempfile

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"):
            link = Path(tmp) / f"clip{ext}"
            link.symlink_to(video.resolve())
            out[ext] = infer_modality(str(link))
    return out


def forward(st: SentenceTransformer, item, prompt: str):
    kwargs = {"prompt": prompt} if prompt else {}
    features = st.preprocess([item], **kwargs)
    with torch.inference_mode():
        out = st(features)
    return features, out["sentence_embedding"][0].numpy()


DEFAULT_FORMS = [
    ("vid-array-with-metadata", BARS, "array+metadata"),
    ("vid-array-no-metadata", BARS, "array"),
    ("vid-decoder-object", SHAPES, "decoder"),
]
DEFAULT_BATCH = [BARS, "h264-1fps-64x48-8s", "vp9-odd-321x179-30fps-3s", "h264-portrait-360x640-30fps-4s"]
DEFAULT_EAGER = [BARS, "h264-330x186-30fps-3s", "hevc-640x360-30fps-4s"]
FORMS = ("array+metadata", "array", "decoder")


def load_cases(path: Path, video_ids: list[str]) -> dict:
    """Rows and check lists from a cases file, in the shapes of the development constants."""
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - {"multimodal", "forms", "queries", "batch", "eager"}
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    known = set(video_ids)

    def check(where: str, ids: list[str]) -> list[str]:
        missing = [v for v in ids if v not in known]
        if missing:
            raise SystemExit(f"{path}: {where}: unknown videos {missing}")
        return list(ids)

    multimodal = []
    for row in data.get("multimodal", []):
        image = row.get("image")
        check(row["id"], row["videos"] + ([image[0]] if image else []))
        if row["prompt"] not in PROMPTS or not row["videos"]:
            raise SystemExit(f"{path}: {row['id']}: needs videos and a known prompt")
        multimodal.append((row["id"], row.get("text"), row["videos"], tuple(image) if image else None, row["prompt"]))
    forms = [(f["id"], check(f["id"], [f["video"]])[0], f["form"]) for f in data.get("forms", [])]
    if any(f[2] not in FORMS for f in forms) or len({f[2] for f in forms}) != len(forms):
        raise SystemExit(f"{path}: each form ({', '.join(FORMS)}) at most once")
    queries = [(q["id"], q["text"]) for q in data.get("queries", [])]
    ids = [r[0] for r in multimodal] + [f[0] for f in forms] + [q[0] for q in queries]
    if len(set(ids)) != len(ids) or set(ids) & known:
        raise SystemExit(f"{path}: row ids must be unique and differ from the video ids")
    return {
        "multimodal": multimodal,
        "forms": forms,
        "queries": queries,
        "batch": check("batch", data.get("batch", [])),
        "eager": check("eager", data.get("eager", [])),
        "quirk_video": video_ids[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON file with the multimodal, decoded-array, query rows and the batch and eager video lists "
        "(default: the development rows, which name development media ids)",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    args.tensors.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.videos / "videos.json").read_text(encoding="utf-8"))
    by_id = {v["id"]: v for v in manifest["videos"]}
    vpath = {k: args.videos / v["file"] for k, v in by_id.items()}
    if args.cases:
        spec = load_cases(args.cases, [v["id"] for v in manifest["videos"]])
    else:
        spec = {
            "multimodal": MULTIMODAL,
            "forms": DEFAULT_FORMS,
            "queries": QUERIES,
            "batch": DEFAULT_BATCH,
            "eager": DEFAULT_EAGER,
            "quirk_video": BARS,
        }

    st = load(args.model_dir)
    model = st[0].model
    processor = st[0].processor
    vproc = processor.video_processor
    assert type(vproc).__name__ == "EmbeddingGemma2VideoProcessor"
    assert (vproc.fps, vproc.max_frames, vproc.max_soft_tokens, vproc.overflow_strategy) == (1, 32, 140, "uniform")
    assert vproc.add_timestamps is False and processor.video_token_id == VIDEO_TOKEN
    tok = processor.tokenizer

    decoded = {k: decode_video(p) for k, p in vpath.items()}
    frames_json = {
        k: {"metadata": d["metadata"], "sampled_indices": d["indices"], "all_frame_sha256": d["all_frame_sha256"]}
        for k, d in decoded.items()
    }
    fixtures.write_json(args.out / "video-frames.json", frames_json)

    def still(video_id: str, index: int) -> Image.Image:
        d = VideoDecoder(str(vpath[video_id]), seek_mode="exact", num_ffmpeg_threads=0)
        return Image.fromarray(d.get_frame_at(index).data.permute(1, 2, 0).numpy())

    cases = [(v["id"], None, [v["id"]], None, "none", "path") for v in manifest["videos"]]
    cases += [(c[0], c[1], c[2], c[3], c[4], "path") for c in spec["multimodal"]]
    cases += [(row_id, None, [video_id], None, "none", form) for row_id, video_id, form in spec["forms"]]
    cases += [(qid, text, [], None, "query", "text") for qid, text in spec["queries"]]

    def all_frames(video_id: str) -> np.ndarray:
        d = VideoDecoder(str(vpath[video_id]), seek_mode="exact", num_ffmpeg_threads=0)
        return d.get_frames_in_range(0, d.metadata.num_frames).data.permute(0, 2, 3, 1).numpy()

    def build_input(text, video_ids, image_src, form):
        if form == "text":
            return text
        if form == "array+metadata":
            m = decoded[video_ids[0]]["metadata"]
            meta = {"fps": m["average_fps"], "total_num_frames": m["num_frames"], "duration": m["duration_seconds"]}
            return {"video": {"array": all_frames(video_ids[0]), "video_metadata": meta}}
        if form == "array":
            return all_frames(video_ids[0])
        if form == "decoder":
            return VideoDecoder(str(vpath[video_ids[0]]), seek_mode="exact", num_ffmpeg_threads=0)
        vids = [str(vpath[v]) for v in video_ids]
        if text is None and image_src is None and len(vids) == 1:
            return video_arg(vpath[video_ids[0]])
        item = {"text": text} if text is not None else {}
        if image_src is not None:
            item["image"] = still(*image_src)
        item["video"] = vids if len(vids) > 1 else vids[0]
        return item

    rows, tensors = [], {}
    for case_id, text, video_ids, image_src, prompt_name, form in cases:
        item = build_input(text, video_ids, image_src, form)
        still_rgb = np.asarray(item["image"]) if isinstance(item, dict) and "image" in item else None
        cap = Capture(model)
        features, emb = forward(st, item, PROMPTS[prompt_name])
        cap.close()
        ids = features["input_ids"][0].tolist()
        arrays: dict[str, np.ndarray] = {
            "input_ids": np.array(ids, dtype=np.int64),
            "token_embeddings": cap.data["token_embeddings"][-1][0].numpy(),
            "embedding": emb,
        }
        if still_rgb is not None:
            arrays["image0_rgb"] = still_rgb
        video_records = []
        if video_ids:
            pv_all = features["pixel_values_videos"]
            pos_all = features["video_position_ids"]
            nfv = [int(x) for x in features["num_frames_per_video"]]
            # Vision hooks: images first (get_image_features), then videos (one call for all frames).
            soft_v = cap.data["soft_tokens"][-1].numpy()
            pooled_v = cap.data["pooled"][-1].numpy()
            enc_v = cap.data["encoder"][-1].numpy()
            frame_off, tok_off = 0, 0
            for k, video_id in enumerate(video_ids):
                d = decoded[video_id]
                frames = d["frames"]
                if form == "array":
                    # Upstream without metadata: every frame kept, then the 32-frame cap.
                    n_all = d["metadata"]["num_frames"]
                    idx = eg2_video.sample_indices(n_all, None, None)
                    frames = torch.from_numpy(all_frames(video_id)[idx]).permute(0, 3, 1, 2).contiguous()
                else:
                    idx = d["indices"]
                r = eg2_video.preprocess(frames)
                t = frames.shape[0]
                if nfv[k] != t:
                    raise SystemExit(f"{case_id}/{video_id}: processor kept {nfv[k]} frames, restated {t}")
                pv = pv_all[frame_off : frame_off + t]
                pos = pos_all[frame_off : frame_off + t]
                if not (torch.equal(r["pixel_values"], pv) and torch.equal(r["positions"], pos)):
                    raise SystemExit(f"{case_id}/{video_id}: restated preprocessing differs from the processor")
                n = r["count"]
                valid = n * POOL * POOL
                soft = soft_v[tok_off : tok_off + t * n]
                pooled = pooled_v[tok_off : tok_off + t * n]
                enc = enc_v[frame_off : frame_off + t, :valid]
                frame_off += t
                tok_off += t * n
                prefix = f"video{k}_"
                arrays.update(
                    {
                        prefix + "frames_uint8": frames.permute(0, 2, 3, 1).numpy(),
                        prefix + "resized_uint8": r["resized"].permute(0, 2, 3, 1).numpy(),
                        prefix + "positions": pos[:, :valid].numpy(),
                        prefix + "pooled": pooled,
                        prefix + "soft_tokens": soft,
                    }
                )
                video_records.append(
                    {
                        "video": video_id,
                        "input_form": form,
                        "metadata": None if form == "array" else d["metadata"],
                        "sampled_indices": idx,
                        "frames": t,
                        "decoded": {"height": int(frames.shape[2]), "width": int(frames.shape[3])},
                        "resized": {"height": r["target"][0], "width": r["target"][1]},
                        "soft_tokens_per_frame": n,
                        "patches_per_frame": valid,
                        "frame_sha256": [sha(f.permute(1, 2, 0)) for f in frames],
                        "digests": {
                            "frames_uint8": sha(arrays[prefix + "frames_uint8"]),
                            "resized_uint8": sha(arrays[prefix + "resized_uint8"]),
                            "pixel_values": sha(pv[:, :valid]),
                            "positions": sha(arrays[prefix + "positions"]),
                            "encoder": sha(enc),
                            "pooled": sha(pooled),
                            "soft_tokens": sha(soft),
                        },
                    }
                )
            if tok_off != soft_v.shape[0]:
                raise SystemExit(f"{case_id}: {soft_v.shape[0]} video soft tokens, {tok_off} expected")
        expected_video_tokens = sum(v["frames"] * v["soft_tokens_per_frame"] for v in video_records)
        if sum(1 for i in ids if i == VIDEO_TOKEN) != expected_video_tokens:
            raise SystemExit(f"{case_id}: <|video|> count differs from frames x soft tokens")
        np.savez(args.tensors / f"{case_id}.npz", **arrays)
        tensors[case_id] = {
            n: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a)} for n, a in arrays.items()
        }
        rows.append(
            {
                "id": case_id,
                "kind": "text" if not video_ids else ("video" if text is None and image_src is None else "multimodal"),
                "input_form": form,
                "prompt": prompt_name,
                "prompt_text": PROMPTS[prompt_name],
                "text": text,
                "image": None if image_src is None else {"video": image_src[0], "frame_index": image_src[1]},
                "videos": video_records,
                "rendered": render(tok, ids),
                "input_ids": ids,
                "tokens": len(ids),
                "video_token_count": int(sum(1 for i in ids if i == VIDEO_TOKEN)),
                "image_token_count": int(sum(1 for i in ids if i == IMAGE_TOKEN)),
                "embedding": fixtures.float32_list(emb),
                "norm": float(np.linalg.norm(emb)),
                "tensors_sha256": {n: v["sha256"] for n, v in tensors[case_id].items()},
            }
        )
        print(case_id, len(ids), [(v["frames"], v["soft_tokens_per_frame"]) for v in video_records], flush=True)

    fixtures.write_jsonl(args.out / "video-embeddings.jsonl", rows)
    fixtures.write_json(args.tensors / "tensors.json", tensors)
    emb_by_id = {r["id"]: np.array(r["embedding"], np.float32) for r in rows}

    reference = {
        "reference": "sentence-transformers 6.1.0 modules, torchcodec 0.17.0 decoding (FFmpeg "
        + tc_core.get_ffmpeg_library_versions()["ffmpeg_version"]
        + "), PyTorch fp32 CPU, sdpa attention, one input per call, full checkpoint",
        "rows": len(rows),
    }
    form_rows = {form: (row_id, video_id) for row_id, video_id, form in spec["forms"]}
    if "array+metadata" in form_rows:
        row_id, video_id = form_rows["array+metadata"]
        reference["path_vs_array_with_metadata_embedding"] = stats(emb_by_id[row_id] - emb_by_id[video_id])
    if "decoder" in form_rows:
        row_id, video_id = form_rows["decoder"]
        reference["path_vs_decoder_object_embedding"] = stats(emb_by_id[row_id] - emb_by_id[video_id])
    if "array" in form_rows:
        row_id, video_id = form_rows["array"]
        reference["path_vs_array_without_metadata"] = {
            "embedding": stats(emb_by_id[row_id] - emb_by_id[video_id]),
            "cosine": cosine(emb_by_id[row_id], emb_by_id[video_id]),
        }
    reference["quirks"] = {
        # sentence-transformers decides by extension on an existing file (a missing file is text).
        "infer_modality_existing_file": modality_by_extension(vpath[spec["quirk_video"]]),
    }
    # Frame-count checks of the sampling (upstream processor on the torchcodec metadata).
    from transformers.video_utils import VideoMetadata

    sampling = {}
    for k, d in decoded.items():
        m = d["metadata"]
        md = VideoMetadata(total_num_frames=m["num_frames"], fps=m["average_fps"], duration=m["duration_seconds"])
        up = vproc.sample_frames(md, fps=1, max_frames=32, overflow_strategy="uniform").tolist()
        sampling[k] = {"upstream": up, "restated_equal": up == d["indices"]}
        if up != d["indices"]:
            raise SystemExit(f"{k}: restated sampling differs from upstream")
    reference["sampling_restated_equals_upstream"] = all(v["restated_equal"] for v in sampling.values())

    video_rows = [r for r in rows if r["kind"] == "video" and r["input_form"] == "path" and r["prompt"] == "none"]
    vids = [r["id"] for r in video_rows]
    table = {qid: {v: float(np.dot(emb_by_id[qid], emb_by_id[v])) for v in vids} for qid, _ in spec["queries"]}
    reference["query_video_cosine"] = table
    reference["query_top1"] = {q: max(t, key=t.get) for q, t in table.items()}

    if not args.skip_stability:
        # Batch of videos vs single calls.
        sub = spec["batch"]
        if sub:
            with torch.inference_mode():
                batched = st.encode([video_arg(vpath[v]) for v in sub], batch_size=4, convert_to_numpy=True)
            reference["batch4_vs_single"] = stats(batched - np.stack([emb_by_id[v] for v in sub]))

        # Restated vision path: padded (1,260 rows) and unpadded, per frame and all frames at once.
        vp = VisionPath(model).eval()
        worst = {"padded_all_frames": 0.0, "unpadded_all_frames": 0.0, "padded_one_frame_at_a_time": 0.0}
        for v in vids:
            z = np.load(args.tensors / f"{v}.npz")
            frames = torch.from_numpy(z["video0_frames_uint8"]).permute(0, 3, 1, 2).contiguous()
            r = eg2_video.preprocess(frames)
            n = r["count"]
            ref = z["video0_soft_tokens"].reshape(frames.shape[0], n, -1)
            with torch.inference_mode():
                sp, _, _ = vp(r["pixel_values"], r["positions"])
                su, _, _ = vp(r["pixel_values"][:, : 9 * n], r["positions"][:, : 9 * n])
                s1 = torch.cat(
                    [
                        vp(r["pixel_values"][i : i + 1], r["positions"][i : i + 1])[0]
                        for i in range(len(r["pixel_values"]))
                    ]
                )
            worst["padded_all_frames"] = max(worst["padded_all_frames"], float(np.abs(sp[:, :n].numpy() - ref).max()))
            worst["unpadded_all_frames"] = max(worst["unpadded_all_frames"], float(np.abs(su.numpy() - ref).max()))
            worst["padded_one_frame_at_a_time"] = max(
                worst["padded_one_frame_at_a_time"], float(np.abs(s1[:, :n].numpy() - ref).max())
            )
        reference["restated_vision_vs_upstream_soft_tokens_max_abs"] = worst
        del vp
        eager = load(args.model_dir, attn="eager")
        e_rows = []
        for v in spec["eager"]:
            _, e = forward(eager, video_arg(vpath[v]), "")
            e_rows.append(e - emb_by_id[v])
        if e_rows:
            reference[f"eager_vs_sdpa_{len(e_rows)}_videos"] = stats(np.stack(e_rows))
        del eager
    reference["nan_or_inf_rows"] = [r["id"] for r in rows if not np.isfinite(np.array(r["embedding"])).all()]
    fixtures.write_json(args.out / "video-reference.json", reference)

    env = fixtures.environment(PACKAGES)
    env["torch_threads"] = torch.get_num_threads()
    env["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    env["torchcodec_ffmpeg"] = tc_core.get_ffmpeg_library_versions()
    fixtures.write_json(
        args.out / "video-environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_video.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "model": {"repo": REPO, "revision": args.revision},
            "assets": [fixtures.file_record(args.model_dir / a, args.model_dir) for a in ASSETS],
            "videos": {k: v for k, v in manifest.items() if k != "videos"}
            | {"files": [{k: v[k] for k in ("id", "file", "sha256")} for v in manifest["videos"]]}
            | ({"cases": fixtures.file_record(args.cases)} if args.cases else {}),
            "environment": env,
        },
    )
    print(json.dumps({k: v for k, v in reference.items() if k not in ("query_video_cosine", "reference")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
