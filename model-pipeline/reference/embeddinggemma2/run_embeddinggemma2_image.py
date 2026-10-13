"""Reference runner for google/embeddinggemma-2, image path (EmbeddingGemma 2 image-path spike). CPU, fp32.

Inputs: a local snapshot pinned by Hub commit SHA and the image set from `make_eg2_images.py` (`images.json`).
Upstream is sentence-transformers 6.1.0 over transformers 5.19.0: `SentenceTransformer.encode` renders every input
through `chat_template.jinja` (prompt as a system message, then text and `<|image|>` placeholders in order), the
`EmbeddingGemma2Processor` expands each placeholder to `<|image>` + n x `<|image|>` + `<image|>` and runs the
`Gemma4ImageProcessor` (torchvision backend: PIL `convert("RGB")`, uint8 bicubic antialiased resize to the
aspect-preserving 48-multiple size within 2,520 patches, x 1/255, 16x16 patches, padding to 2,520), then the model
scatters the vision soft tokens into the text sequence and mean-pools every token (prompt and image tokens included).

Writes into `--out` (small; fixture candidates):

- image-embeddings.jsonl: one row per input (image only, image with a prompt, interleaved text and images, text
  queries for a retrieval sanity check): the rendered ids, per image the decoded and resized sizes, the soft-token
  count and SHA-256 digests of every intermediate tensor, and the fp32 embedding.
- image-preprocess.json: per image, the processor restated step by step (must equal the processor bit for bit),
  and the transformers PIL backend (`Gemma4ImageProcessorPil`, PIL bicubic) against the torchvision backend: pixel
  differences and the embedding difference they cause.
- image-reference.json: stability of the reference (batched vs single, eager vs sdpa attention, text-and-vision
  load vs full load, restated vision path vs upstream, unpadded vs padded patches, encode end to end vs the module
  forward), and a query-by-image cosine table for the retrieval sanity check.
- image-environment.json: interpreter, packages, asset and image digests.

Writes into `--tensors` (large; never committed): one `.npz` per row with the resized uint8 image, the pixel and
position tensors (valid rows), vision encoder output (valid rows), pooled soft tokens (768), projected soft tokens
(512), token embeddings and the embedding, and `tensors.json` with their SHA-256 digests.

Usage:
    python run_embeddinggemma2_image.py --model-dir <snapshot> --revision <sha> --images <dir> --out <dir>
        --tensors <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from eg2_vision import VisionPath, patchify
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision.transforms.v2 import functional as tvF
from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

from model_pipeline import fixtures

REPO = "google/embeddinggemma-2"
PROMPTS = {"none": "", "query": "task: search result | query: ", "document": "title: none | text: "}
IMAGE_TOKEN, BOI, EOI = 258880, 255999, 258882
MAX_PATCHES, PATCH, POOL = 2520, 16, 3
# Interleaved inputs: (id, text with placeholders or None, image ids, prompt name).
INTERLEAVED = [
    ("mm-caption-placeholder", "A cat resting on a blanket. <|image|>", ["photo-chelsea"], "none"),
    ("mm-two-images", "<|image|> next to <|image|>", ["photo-coffee", "photo-rocket"], "none"),
    ("mm-text-then-image-no-placeholder", "Product photo:", ["synth-shapes"], "none"),
    ("mm-document-prompt", "A cup of coffee on a wooden table. <|image|>", ["photo-coffee"], "document"),
    ("img-query-prompt", None, ["photo-chelsea"], "query"),
    ("img-document-prompt", None, ["photo-astronaut"], "document"),
]
QUERIES = [
    ("q-cat", "a photo of a cat"),
    ("q-coffee", "a cup of coffee"),
    ("q-rocket", "a rocket launch"),
    ("q-astronaut", "portrait of an astronaut"),
    ("q-galaxies", "deep space full of galaxies"),
    ("q-cameraman", "black and white photo of a man with a camera"),
    ("q-shapes", "simple colored geometric shapes"),
    ("q-text", "an image of printed text"),
]
PACKAGES = ["torch", "torchvision", "transformers", "sentence-transformers", "tokenizers", "numpy", "pillow"]
ASSETS = ["config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json", "model.safetensors"]


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(diff: np.ndarray) -> dict:
    a = np.abs(np.asarray(diff, dtype=np.float64)).ravel()
    if a.size == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    return {"max_abs": float(a.max()), "mean_abs": float(a.mean())}


def load(model_dir: Path, attn: str | None = None, full: bool = True) -> SentenceTransformer:
    model_kwargs = {"dtype": torch.float32}
    if attn:
        model_kwargs["attn_implementation"] = attn
    config_kwargs = None if full else {"audio_config": None}
    st = SentenceTransformer(str(model_dir), device="cpu", model_kwargs=model_kwargs, config_kwargs=config_kwargs)
    st.eval()
    return st


def restate_preprocess(image: Image.Image) -> dict:
    """The torchvision-backend processor, step by step, for one image (no padding across images)."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    chw = tvF.pil_to_tensor(rgb)  # uint8 [3, H, W]
    h, w = chw.shape[-2:]
    th, tw = get_aspect_ratio_preserving_size(h, w, PATCH, MAX_PATCHES, POOL)
    resized = (
        chw
        if (th, tw) == (h, w)
        else tvF.resize(chw, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    )
    rescaled = resized * 0.00392156862745098  # processor_config rescale_factor (1/255), float32 result
    patches, pos, count = patchify(rescaled, MAX_PATCHES, PATCH)
    return {
        "rgb": np.asarray(rgb),
        "resized": resized.permute(1, 2, 0).numpy(),
        "target": (th, tw),
        "pixel_values": patches,
        "positions": pos,
        "count": count,
    }


class Capture:
    """Forward hooks on the vision encoder, the vision tower, embed_vision and the language model."""

    def __init__(self, model: torch.nn.Module):
        self.data: dict[str, torch.Tensor] = {}
        self.handles = []
        if model.vision_tower is not None:
            self._hook(model.vision_tower.encoder, "encoder")
            self._hook(model.vision_tower, "pooled")
            self._hook(model.embed_vision, "soft_tokens")
        self._hook(model.language_model, "token_embeddings")

    def _hook(self, module, name):
        def fn(_m, _i, out):
            t = out if isinstance(out, torch.Tensor) else out.last_hidden_state
            self.data[name] = t.detach().clone()

        self.handles.append(module.register_forward_hook(fn))

    def close(self):
        for h in self.handles:
            h.remove()


def render(tok, ids: list[int]) -> str:
    """The decoded sequence with each run of `<|image|>` written once with its length, e.g. `<|image|>x260`."""
    parts, run, chunk = [], 0, []
    for i in [*ids, None]:
        if i == IMAGE_TOKEN:
            if chunk:
                parts.append(tok.decode(chunk, skip_special_tokens=False))
                chunk = []
            run += 1
            continue
        if run:
            parts.append(f"<|image|>x{run}")
            run = 0
        if i is not None:
            chunk.append(i)
    if chunk:
        parts.append(tok.decode(chunk, skip_special_tokens=False))
    return "".join(parts)


def st_input(text: str | None, images: list[Image.Image]):
    if text is None and len(images) == 1:
        return images[0]
    if text is None:
        raise ValueError("several images need a text with placeholders")
    return {"text": text, "image": images if len(images) > 1 else images[0]} if images else text


def forward(st: SentenceTransformer, item, prompt: str, override_pixels: torch.Tensor | None = None):
    kwargs = {"prompt": prompt} if prompt else {}
    features = st.preprocess([item], **kwargs)
    if override_pixels is not None:
        features["pixel_values"] = override_pixels
    with torch.inference_mode():
        out = st(features)
    return features, out["sentence_embedding"][0].numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    args.tensors.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.images / "images.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in manifest["images"]}

    def open_image(image_id: str) -> Image.Image:
        return Image.open(args.images / by_id[image_id]["file"])

    st = load(args.model_dir)
    model = st[0].model
    processor = st[0].processor
    assert type(processor.image_processor).__name__ == "Gemma4ImageProcessor"
    assert processor.image_processor.backend == "torchvision"
    tok = processor.tokenizer

    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

    ip_cfg = json.loads((args.model_dir / "processor_config.json").read_text())["image_processor"]
    pil_backend = Gemma4ImageProcessorPil(**{k: v for k, v in ip_cfg.items() if k != "image_processor_type"})

    rows, tensors, preprocess = [], {}, {}
    cases = [(r["id"], None, [r["id"]], "none") for r in manifest["images"]]
    cases += INTERLEAVED
    cases += [(qid, text, [], "query") for qid, text in QUERIES]

    for case_id, text, image_ids, prompt_name in cases:
        images = [open_image(i) for i in image_ids]
        cap = Capture(model)
        features, emb = forward(st, st_input(text, images), PROMPTS[prompt_name])
        cap.close()
        ids = features["input_ids"][0].tolist()
        tensor_file = args.tensors / f"{case_id}.npz"
        arrays: dict[str, np.ndarray] = {
            "input_ids": np.array(ids, dtype=np.int64),
            "token_embeddings": cap.data["token_embeddings"][0].numpy(),
            "embedding": emb,
        }
        image_records = []
        if images:
            pv = features["pixel_values"].numpy()
            pos = features["image_position_ids"].numpy()
            enc = cap.data["encoder"].numpy()
            pooled = cap.data["pooled"].numpy()
            soft = cap.data["soft_tokens"].numpy()
            offset = 0
            for k, (image_id, image) in enumerate(zip(image_ids, images, strict=True)):
                r = restate_preprocess(image)
                n = r["count"]
                valid = n * POOL * POOL
                same_pixels = bool(np.array_equal(r["pixel_values"].numpy(), pv[k]))
                same_pos = bool(np.array_equal(r["positions"].numpy(), pos[k]))
                if not (same_pixels and same_pos):
                    raise SystemExit(f"{case_id}/{image_id}: restated preprocessing differs from the processor")
                prefix = f"image{k}_"
                arrays.update(
                    {
                        prefix + "resized_uint8": r["resized"],
                        prefix + "pixel_values": pv[k, :valid],
                        prefix + "positions": pos[k, :valid],
                        prefix + "encoder": enc[k, :valid],
                        prefix + "pooled": pooled[offset : offset + n],
                        prefix + "soft_tokens": soft[offset : offset + n],
                    }
                )
                offset += n
                image_records.append(
                    {
                        "image": image_id,
                        "decoded": {"mode": image.mode, "width": image.width, "height": image.height},
                        "rgb_sha256": sha(r["rgb"]),
                        "resized": {"height": r["target"][0], "width": r["target"][1]},
                        "soft_tokens": n,
                        "patches": valid,
                        "digests": {
                            name: sha(arrays[prefix + name])
                            for name in (
                                "resized_uint8",
                                "pixel_values",
                                "positions",
                                "encoder",
                                "pooled",
                                "soft_tokens",
                            )
                        },
                    }
                )
                if image_id not in preprocess and text is None and prompt_name == "none":
                    pil = pil_backend(images=[image], return_tensors="pt")
                    pil_pv = pil["pixel_values"]
                    diff = (pil_pv[0, :valid] - pv[k, :valid]).numpy() * 255.0
                    _, emb_pil = forward(st, image, "", override_pixels=pil_pv)
                    preprocess[image_id] = {
                        "decoded": {"mode": image.mode, "width": image.width, "height": image.height},
                        "resized": {"height": r["target"][0], "width": r["target"][1]},
                        "resize_skipped": tuple(r["target"]) == (image.height, image.width),
                        "soft_tokens": n,
                        "restated_equals_processor": True,
                        "pil_backend_vs_torchvision": {
                            "pixel_levels_max_abs": float(np.abs(diff).max()),
                            "pixel_levels_differing_fraction": float((np.abs(diff) > 0.5).mean()),
                            "embedding": stats(emb_pil - emb),
                            "cosine": float(np.dot(emb_pil, emb) / np.linalg.norm(emb_pil) / np.linalg.norm(emb)),
                        },
                    }
        np.savez(tensor_file, **arrays)
        tensors[case_id] = {
            name: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a)} for name, a in arrays.items()
        }
        rows.append(
            {
                "id": case_id,
                "kind": "text" if not images else ("image" if text is None else "interleaved"),
                "prompt": prompt_name,
                "prompt_text": PROMPTS[prompt_name],
                "text": text,
                "images": image_records,
                "rendered": render(tok, ids),
                "input_ids": ids,
                "tokens": len(ids),
                "image_token_count": int(sum(1 for i in ids if i == IMAGE_TOKEN)),
                "embedding": fixtures.float32_list(emb),
                "norm": float(np.linalg.norm(emb)),
                "tensors_sha256": {name: v["sha256"] for name, v in tensors[case_id].items()},
            }
        )
        print(case_id, len(ids), [r["soft_tokens"] for r in image_records], flush=True)

    fixtures.write_jsonl(args.out / "image-embeddings.jsonl", rows)
    fixtures.write_json(args.out / "image-preprocess.json", preprocess)
    fixtures.write_json(args.tensors / "tensors.json", tensors)

    # Reference stability.
    emb_by_id = {r["id"]: np.array(r["embedding"], dtype=np.float32) for r in rows}
    image_ids = [r["id"] for r in manifest["images"]]
    imgs = [open_image(i) for i in image_ids]
    with torch.inference_mode():
        batched = st.encode(imgs, batch_size=8, convert_to_numpy=True)
        e2e = np.stack([st.encode([im], batch_size=1, convert_to_numpy=True)[0] for im in imgs])
    singles = np.stack([emb_by_id[i] for i in image_ids])
    reference = {
        "reference": "sentence-transformers 6.1.0 modules on PIL images opened from the files (Image.open), PyTorch "
        "fp32 CPU, sdpa attention, one input per call, full checkpoint",
        "rows": len(rows),
        "batched8_vs_single": stats(batched - singles),
        "encode_end_to_end_vs_forward": stats(e2e - singles),
    }

    # Restated vision path vs upstream, padded and unpadded.
    vp = VisionPath(model).eval()
    worst_padded, worst_unpadded, worst_pooled = 0.0, 0.0, 0.0
    for image_id in image_ids:
        npz = np.load(args.tensors / f"{image_id}.npz")
        pv = torch.from_numpy(npz["image0_pixel_values"])
        pos = torch.from_numpy(npz["image0_positions"])
        n = pv.shape[0] // 9
        pad = MAX_PATCHES - pv.shape[0]
        pv_p = torch.nn.functional.pad(pv, (0, 0, 0, pad))[None]
        pos_p = torch.nn.functional.pad(pos, (0, 0, 0, pad), value=-1)[None]
        with torch.inference_mode():
            soft_p, pooled_p, valid_p = vp(pv_p, pos_p)
            soft_u, _, valid_u = vp(pv[None], pos[None])
        assert int(valid_p.sum()) == n and bool(valid_p[0, :n].all()) and int(valid_u.sum()) == n
        worst_padded = max(worst_padded, float((soft_p[0, :n].numpy() - npz["image0_soft_tokens"]).__abs__().max()))
        worst_pooled = max(worst_pooled, float((pooled_p[0, :n].numpy() - npz["image0_pooled"]).__abs__().max()))
        worst_unpadded = max(worst_unpadded, float((soft_u[0].numpy() - npz["image0_soft_tokens"]).__abs__().max()))
    reference["restated_vision_padded_vs_upstream_soft_tokens_max_abs"] = worst_padded
    reference["restated_vision_padded_vs_upstream_pooled_max_abs"] = worst_pooled
    reference["restated_vision_unpadded_vs_upstream_soft_tokens_max_abs"] = worst_unpadded

    # Query-by-image cosine table (retrieval sanity).
    photo_ids = [i for i in image_ids if i.startswith(("photo-", "synth-shapes", "synth-text"))]
    table = {qid: {pid: float(np.dot(emb_by_id[qid], emb_by_id[pid])) for pid in photo_ids} for qid, _ in QUERIES}
    reference["query_image_cosine"] = table
    reference["query_top1"] = {qid: max(v, key=v.get) for qid, v in table.items()}
    del st, model, vp

    sub = image_ids[::3] + ["mm-two-images"]
    eager = load(args.model_dir, attn="eager")
    eager_rows = []
    for case_id in sub:
        case = next(c for c in cases if c[0] == case_id)
        _, e = forward(eager, st_input(case[1], [open_image(i) for i in case[2]]), PROMPTS[case[3]])
        eager_rows.append(e - emb_by_id[case_id])
    reference["eager_vs_sdpa_every_third_image"] = stats(np.stack(eager_rows))
    del eager
    partial = load(args.model_dir, full=False)
    part_rows = []
    for case_id in sub:
        case = next(c for c in cases if c[0] == case_id)
        _, e = forward(partial, st_input(case[1], [open_image(i) for i in case[2]]), PROMPTS[case[3]])
        part_rows.append(e - emb_by_id[case_id])
    reference["text_and_vision_load_vs_full_every_third_image"] = stats(np.stack(part_rows))
    del partial
    reference["nan_or_inf_rows"] = [r["id"] for r in rows if not np.isfinite(np.array(r["embedding"])).all()]
    fixtures.write_json(args.out / "image-reference.json", reference)

    env = fixtures.environment(PACKAGES)
    env["torch_threads"] = torch.get_num_threads()
    env["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    fixtures.write_json(
        args.out / "image-environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_image.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "model": {"repo": REPO, "revision": args.revision},
            "assets": [fixtures.file_record(args.model_dir / a, args.model_dir) for a in ASSETS],
            "images": {k: v for k, v in manifest.items() if k != "images"}
            | {"files": [{k: r[k] for k in ("id", "file", "sha256", "decoded_sha256")} for r in manifest["images"]]},
            "environment": env,
        },
    )
    print(json.dumps({k: v for k, v in reference.items() if k not in ("query_image_cosine", "reference")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
