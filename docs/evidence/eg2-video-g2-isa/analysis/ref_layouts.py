"""The reference's own soft tokens under other legitimate PyTorch batch layouts (the runner's restated vision path),
run through the released ONNX multimodal text graph, compared with the reference token embeddings.

usage: ref_layouts.py <reference dir (model-pipeline/reference/embeddinggemma2)> <hf snapshot> <frames npz dir>
       <hosted refs dir> <text-mm.onnx> <video-embeddings.jsonl> <out json> [rows...]
"""

import json
import sys

ref_dir, snap, frames_dir, refs_dir, mm, jsonl, out_path, *row_ids = sys.argv[1:]
sys.path.insert(0, ref_dir)

import eg2_video  # noqa: E402

eg2_video.pin_torchcodec_ffmpeg()
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from eg2_vision import VisionPath  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

torch.set_num_threads(6)
row_ids = row_ids or ["commons-video-04"]
rows = {json.loads(line)["id"]: json.loads(line) for line in open(jsonl, encoding="utf-8")}
st = SentenceTransformer(snap, device="cpu", model_kwargs={"dtype": torch.float32})
st.eval()
vp = VisionPath(st[0].model).eval()
so = ort.SessionOptions()
so.intra_op_num_threads = 6
so.enable_cpu_mem_arena = False
sess = ort.InferenceSession(mm, so, providers=["CPUExecutionProvider"])
result = {}
for rid in row_ids:
    z = np.load(f"{frames_dir}/{rid}.npz")
    hz = np.load(f"{refs_dir}/{rid}.npz")
    ref_soft = hz["video0_soft_tokens"]
    ref_te = hz["token_embeddings"]
    assert np.array_equal(z["video0_soft_tokens"], ref_soft) and np.array_equal(z["token_embeddings"], ref_te)
    frames = torch.from_numpy(z["video0_frames_uint8"]).permute(0, 3, 1, 2).contiguous()
    r = eg2_video.preprocess(frames)
    n = r["count"]
    with torch.inference_mode():
        sp, _, _ = vp(r["pixel_values"], r["positions"])
        su, _, _ = vp(r["pixel_values"][:, : 9 * n], r["positions"][:, : 9 * n])
        s1 = torch.cat([vp(r["pixel_values"][i : i + 1], r["positions"][i : i + 1])[0] for i in range(len(r["pixel_values"]))])
    layouts = {
        "reference": ref_soft,
        "padded_all_frames": sp[:, :n].reshape(-1, 512).numpy(),
        "unpadded_all_frames": su.reshape(-1, 512).numpy(),
        "padded_one_frame_at_a_time": s1[:, :n].reshape(-1, 512).numpy(),
    }
    ids = np.array(rows[rid]["input_ids"], dtype=np.int64)
    vals, counts = np.unique(ids, return_counts=True)
    mask = ids == vals[np.argmax(counts)]
    result[rid] = {}
    for name, soft in layouts.items():
        emb = np.zeros((1, ids.size, 512), dtype=np.float32)
        emb[0, mask] = soft
        te = sess.run(
            ["token_embeddings"],
            {"input_ids": ids[None], "attention_mask": np.ones((1, ids.size), np.int64), "multimodal_embeds": emb, "multimodal_mask": mask[None]},
        )[0][0]
        result[rid][name] = {
            "soft_tokens_row_relative": float(np.abs(soft.astype(np.float64) - ref_soft).max() / np.abs(ref_soft).max()),
            "token_embeddings_row_relative": float(np.abs(te.astype(np.float64) - ref_te).max() / np.abs(ref_te).max()),
        }
        print(rid, name, json.dumps(result[rid][name]), flush=True)
json.dump(result, open(out_path, "w"), indent=1)
