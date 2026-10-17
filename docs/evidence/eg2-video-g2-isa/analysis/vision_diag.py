"""EmbeddingGemma 2 vision graph on the sampled frames of one video, under ORT session variants, plus a per-node digest
dump of the unoptimized graph for one frame (to find the first node whose output depends on the CPU's ISA).

usage: vision_diag.py <reference dir> <video> <vision.onnx> <out dir> [--frames k] [--threads n]
       [--variants all,extended,basic,disable] [--dump-nodes N]
Run with the pinned FFmpeg 8 on LD_LIBRARY_PATH.
"""

import argparse
import hashlib
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("ref_dir")
p.add_argument("video")
p.add_argument("vision")
p.add_argument("out")
p.add_argument("--frames", type=int, default=4)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--variants", default="all,disable")
p.add_argument("--dump-nodes", type=int, default=0)
a = p.parse_args()
sys.path.insert(0, a.ref_dir)
import eg2_video  # noqa: E402

eg2_video.pin_torchcodec_ffmpeg()
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


os.makedirs(a.out, exist_ok=True)
d = VideoDecoder(a.video, seek_mode="exact", num_ffmpeg_threads=0)
m = d.metadata
idx = eg2_video.sample_indices(m.num_frames, m.average_fps, m.duration_seconds)[: a.frames]
frames = d.get_frames_at(indices=idx).data.contiguous()
r = eg2_video.preprocess(frames)
pv = r["pixel_values"].numpy().astype(np.float32)
pos = r["positions"].numpy().astype(np.int64)
n = r["count"]
summary = {"frames": idx, "count": n, "pixel_values_sha256": sha(pv), "variants": {}}
print("pixel values", summary["pixel_values_sha256"], pv.shape, flush=True)


def session(model, level):
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = LEVELS[level]
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


for v in [x for x in a.variants.split(",") if x]:
    s = session(a.vision, v)
    t = time.time()
    soft = np.concatenate([s.run(["soft_tokens"], {"pixel_values": pv[i : i + 1], "pixel_position_ids": pos[i : i + 1]})[0][0, :n] for i in range(len(pv))])
    summary["variants"][v] = {"sha256": sha(soft), "per_frame": [sha(soft[i * n : (i + 1) * n]) for i in range(len(pv))], "seconds": round(time.time() - t, 1)}
    np.save(os.path.join(a.out, f"soft-{v}.npy"), soft)
    print(v, json.dumps(summary["variants"][v]), flush=True)
    del s

if a.dump_nodes:
    import onnx

    model = onnx.load(a.vision)
    existing = {o.name for o in model.graph.output}
    names = []
    for node in model.graph.node[: a.dump_nodes]:
        for o in node.output:
            if o and o not in existing:
                names.append((node.name, node.op_type, o))
                model.graph.output.append(onnx.ValueInfoProto(name=o))
    path = os.path.join(a.out, "vision-dump.onnx")
    onnx.save(model, path)
    del model
    s = session(path, "disable")
    outs = s.run([o for _, _, o in names], {"pixel_values": pv[:1], "pixel_position_ids": pos[:1]})
    os.remove(path)
    rows = []
    for (nn, op, o), val in zip(names, outs):
        val = np.asarray(val)
        rows.append({"node": nn, "op": op, "output": o, "sha256": sha(val), "dtype": str(val.dtype), "shape": list(val.shape)})
        if val.dtype == np.float32 and val.size < 6_000_000:
            np.save(os.path.join(a.out, f"node-{len(rows) - 1:04d}.npy"), val)
    json.dump(rows, open(os.path.join(a.out, "nodes.json"), "w"), indent=0)
    print("dumped", len(rows), "node outputs", flush=True)
json.dump(summary, open(os.path.join(a.out, "vision.json"), "w"), indent=1)
