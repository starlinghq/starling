"""Re-run the PyTorch reference (the runner's own load/forward/Capture) on one release video on this machine and compare
its soft tokens, token embeddings and embedding with the accepted reference tensors.

usage: ref_rerun.py <reference dir> <hf snapshot> <video> <hosted refs dir> <video-embeddings.jsonl> <out json> [threads]
Run with the pinned FFmpeg 8 on LD_LIBRARY_PATH.
"""

import json
import os
import sys
from pathlib import Path

ref_dir, snap, video, refs_dir, jsonl, out_path, *rest = sys.argv[1:]
threads = int(rest[0]) if rest else 6
sys.path.insert(0, ref_dir)
import run_embeddinggemma2_video as runner  # noqa: E402  (pins torchcodec's FFmpeg on import)
import numpy as np  # noqa: E402
import torch  # noqa: E402

if threads:
    torch.set_num_threads(threads)
rid = Path(video).stem
row = next(json.loads(x) for x in open(jsonl, encoding="utf-8") if json.loads(x)["id"] == rid)
hz = np.load(os.path.join(refs_dir, rid + ".npz"))
st = runner.load(Path(snap))
model = st[0].model
cap = runner.Capture(model)
feats, emb = runner.forward(st, runner.video_arg(Path(video)), "")
cap.close()
soft = torch.cat([t.reshape(-1, t.shape[-1]) for t in cap.data["soft_tokens"]]).numpy()
te = cap.data["token_embeddings"][0][0].numpy()
ref_soft, ref_te = hz["video0_soft_tokens"], hz["token_embeddings"]
ref_emb = np.array(row["embedding"], dtype=np.float32)
ids = feats["input_ids"][0].tolist() if "input_ids" in feats else None
res = {
    "row": rid,
    "torch_threads": torch.get_num_threads(),
    "ids_equal": ids == row["input_ids"] if ids is not None else None,
    "soft_shape": list(soft.shape),
    "soft_tokens_row_relative": float(np.abs(soft.astype(np.float64) - ref_soft).max() / np.abs(ref_soft).max()) if soft.shape == ref_soft.shape else None,
    "token_embeddings_row_relative": float(np.abs(te.astype(np.float64) - ref_te).max() / np.abs(ref_te).max()) if te.shape == ref_te.shape else None,
    "embedding_max_abs": float(np.abs(emb - ref_emb).max()),
    "soft_bit_identical": bool(np.array_equal(soft, ref_soft)),
    "te_bit_identical": bool(np.array_equal(te, ref_te)),
}
print(json.dumps(res), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
np.save(out_path.replace(".json", "-soft.npy"), soft)
