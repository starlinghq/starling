"""Exportable image path of EmbeddingGemma 2 (transformers 5.19.0; image-path spike).

Upstream (`EmbeddingGemma2Model.get_image_features` and `Gemma4VisionModel.forward`) maps the processor's
`pixel_values` ([batch, 2520, 768]: 16x16x3 patches in [0, 1], zero-padded to 2,520 rows) and `image_position_ids`
([batch, 2520, 2]: patch (x, y), padding rows (-1, -1)) to soft tokens:

1. patch embedder: `input_proj(2 * (pixel - 0.5))` (768 -> 768, no bias) plus the sum of a learned x and y position
   embedding (tables of 10,240 rows); padding rows get no position embedding,
2. 16 bidirectional encoder layers (hidden 768, 12 heads of 64, Q/K RMSNorm with scale, V RMSNorm without, 2-D
   axial RoPE with theta 100 over (x, y), attention scaling 1.0, gated GELU-tanh MLP 3,072), padding rows masked as
   keys,
3. pooler: padding rows zeroed, then each 3x3 block of patches averaged (block index
   `x // 3 + (width / 3) * (y // 3)`), times sqrt(768); `standardize` is false in this checkpoint,
4. `embed_vision`: RMSNorm without scale (eps 1e-6), then `Linear(768, 512, bias=False)`,
5. padding soft tokens stripped (upstream boolean indexing). The valid soft tokens are always the first
   `(H / 48) * (W / 48)` of the 280 slots, in row-major block order.

`VisionPath` restates 1 to 4 with fixed-shape outputs: `soft_tokens` ([batch, P / 9, 512], what the text model
receives), `pooled` ([batch, P / 9, 768], before `embed_vision`; diagnostic) and `soft_token_mask` ([batch, P / 9],
true for valid slots). Step 5 is the caller's: take the first `count` rows. The pooling kernel is fixed at 3 (the
checkpoint's `pooling_kernel_size`), so `P` (2,520 upstream) may be any multiple of 9 that holds the image's patches:
an unpadded input (`P = 9 * count`) gives the same soft tokens.

The text model takes the soft tokens in place of the token embedding at `<|image|>` positions (unscaled; text tokens
are `embed_tokens(id) * sqrt(512)`), and computes the per-layer embeddings from that mixed sequence.
`add_multimodal_inputs` rewrites the released text graph to accept them (see its docstring).
"""

from __future__ import annotations

import torch
from torch import nn

POOL = 3


class VisionPath(nn.Module):
    def __init__(self, model: nn.Module):
        """`model` is `EmbeddingGemma2Model` (the sentence-transformers module's `model`)."""
        super().__init__()
        self.tower = model.vision_tower
        self.embed = model.embed_vision
        if self.tower.config.pooling_kernel_size != POOL or self.tower.config.standardize:
            raise ValueError("this restatement assumes pooling_kernel_size 3 and standardize false")
        self.root = float(self.tower.config.hidden_size) ** 0.5

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor):
        tower = self.tower
        padding = (pixel_position_ids == -1).all(dim=-1)  # [B, P]
        hidden = tower.patch_embedder(pixel_values, pixel_position_ids, padding)
        neg = torch.finfo(hidden.dtype).min
        zero = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        low = torch.full((), neg, dtype=hidden.dtype, device=hidden.device)
        seq = pixel_values.shape[1]
        mask = torch.where(padding.logical_not()[:, None, None, :], zero, low).expand(-1, -1, seq, -1)
        cos_sin = tower.encoder.rotary_emb(hidden, pixel_position_ids)
        for layer in tower.encoder.layers:
            hidden = layer(hidden, attention_mask=mask, position_embeddings=cos_sin, position_ids=pixel_position_ids)

        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        length = seq // (POOL * POOL)
        pos = pixel_position_ids.clamp(min=0)
        max_x = pos[..., 0].max(dim=-1, keepdim=True)[0] + 1
        block = torch.div(pos, POOL, rounding_mode="floor")
        index = block[..., 0] + torch.div(max_x, POOL, rounding_mode="floor") * block[..., 1]  # [B, P]
        slots = torch.arange(length, device=hidden.device)
        weights = (index.unsqueeze(-1) == slots).to(hidden.dtype) / float(POOL * POOL)  # [B, P, L]
        pooled = weights.transpose(1, 2) @ hidden  # [B, L, 768]
        valid = (weights != 0).any(dim=1)  # [B, L]
        pooled = pooled * self.root
        return self.embed(pooled), pooled, valid


def load_model(model_dir: str, attn: str = "sdpa") -> nn.Module:
    """`EmbeddingGemma2Model` with text and vision towers (no audio), fp32, eval."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32, attn_implementation=attn, audio_config=None)
    model.eval()
    return model


def patchify(image_chw_float: torch.Tensor, max_patches: int = 2520, patch: int = 16):
    """Processor steps 3 to 6 for one resized, rescaled image ([3, H, W] float): patches, (x, y) positions, padding."""
    c, h, w = image_chw_float.shape
    ph, pw = h // patch, w // patch
    patches = image_chw_float.reshape(c, ph, patch, pw, patch).permute(1, 3, 2, 4, 0).reshape(ph * pw, -1)
    gx, gy = torch.meshgrid(torch.arange(pw), torch.arange(ph), indexing="xy")
    pos = torch.stack([gx, gy], dim=-1).reshape(ph * pw, 2)
    pad = max_patches - patches.shape[0]
    if pad > 0:
        patches = torch.nn.functional.pad(patches, (0, 0, 0, pad))
        pos = torch.nn.functional.pad(pos, (0, 0, 0, pad), value=-1)
    return patches, pos, (ph * pw) // (POOL * POOL)


def add_multimodal_inputs(graph, image_dtype: int | None = None):
    """Rewrite the released text graph (`embeddinggemma2-text-fp32.onnx`) in place to accept soft tokens.

    Adds two inputs, `multimodal_embeds` (float32, [batch, sequence, 512]) and `multimodal_mask` (bool,
    [batch, sequence]), and one `Where` node after the scaled token embedding (`Gather` on `input_ids`, then `Mul` by
    `embed_scale`): where the mask is true the sequence takes the soft token, elsewhere the scaled token embedding.
    Every consumer of the scaled embedding (the first layer's input norm, the per-layer-embedding projection, the
    residual) is rewired to the `Where` output, so the per-layer embeddings see the mixed sequence as upstream does.
    Weights, node order and every other node are unchanged. Returns the name of the rewired tensor.
    """
    from onnx import TensorProto, helper

    g = graph.graph if hasattr(graph, "graph") else graph
    gathers = [n for n in g.node if n.op_type == "Gather" and "input_ids" in n.input]
    if len(gathers) != 1:
        raise ValueError(f"expected one Gather on input_ids, found {len(gathers)}")
    emb = gathers[0].output[0]
    muls = [n for n in g.node if n.op_type == "Mul" and emb in n.input]
    if len(muls) != 1:
        raise ValueError(f"expected one Mul on the token embedding, found {len(muls)}")
    scaled = muls[0].output[0]
    g.input.append(helper.make_tensor_value_info("multimodal_embeds", TensorProto.FLOAT, ["batch", "sequence", 512]))
    g.input.append(helper.make_tensor_value_info("multimodal_mask", TensorProto.BOOL, ["batch", "sequence"]))
    axes = helper.make_tensor("mm_unsqueeze_axes", TensorProto.INT64, [1], [-1])
    g.initializer.append(axes)
    mixed = "inputs_embeds_mixed"
    nodes = [
        helper.make_node("Unsqueeze", ["multimodal_mask", "mm_unsqueeze_axes"], ["mm_mask_3d"], name="mm_unsqueeze"),
        helper.make_node("Where", ["mm_mask_3d", "multimodal_embeds", scaled], [mixed], name="mm_where"),
    ]
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == scaled:
                n.input[i] = mixed
    index = list(g.node).index(muls[0]) + 1
    for k, n in enumerate(nodes):
        g.node.insert(index + k, n)
    return mixed


def add_multimodal_inputs_release(graph) -> dict:
    """Rewrite the released text graph in place for the `embeddinggemma-2-vision-onnx-r1` multimodal asset.

    Same inputs as `add_multimodal_inputs` (`multimodal_embeds` float32 [batch, sequence, 512], `multimodal_mask`
    bool [batch, sequence]), but the released nodes keep their exact inputs, so ONNX Runtime fuses them exactly as in
    the released graph and a text-only call (mask all false) gives the released graph's outputs bit for bit.

    Why: the spike's single `Where` after `Mul(embedding, embed_scale)` sat between that `Mul` and the per-layer
    embedding (PLE) projection `MatMul(scaled, W) * (1 / sqrt(512))`. In the released graph ONNX Runtime's
    `MatMulScaleFusion` (an extended-level optimization) folds both scalars into one `FusedMatMul(embedding, W,
    alpha = sqrt(512) / sqrt(512) = 0.99999994)`; with the `Where` in between it can only fold the second one
    (`FusedMatMul(scaled, W, alpha = 0.0442)`), so the PLE input is rounded differently: about 1e-3 abs on token
    embeddings of about 2,000 (5e-7 relative) on text-only input. With graph optimizations disabled or at the basic
    level the two graphs were identical.

    Layout here:

    - the scaled-embedding consumers that are not the PLE projection (first layer's input norm, residual) read
      `Where(mask, multimodal_embeds, scaled)`;
    - the PLE projection keeps its released input (`scaled`); its output `Mul(MatMul(scaled, W), s)` is mixed after
      the fact with a second projection of the soft tokens using the same weight and scale initializers:
      `Where(mask, Mul(MatMul(multimodal_embeds, W), s), released)`. The PLE projection is position-wise, so this
      equals projecting the mixed sequence.

    Returns the names of the added tensors.
    """
    from onnx import TensorProto, helper

    g = graph.graph if hasattr(graph, "graph") else graph
    gathers = [n for n in g.node if n.op_type == "Gather" and "input_ids" in n.input]
    if len(gathers) != 1:
        raise ValueError(f"expected one Gather on input_ids, found {len(gathers)}")
    emb = gathers[0].output[0]
    muls = [n for n in g.node if n.op_type == "Mul" and emb in n.input]
    if len(muls) != 1:
        raise ValueError(f"expected one Mul on the token embedding, found {len(muls)}")
    scaled = muls[0].output[0]
    consumers = [n for n in g.node if scaled in n.input]
    ple_mm = [n for n in consumers if n.op_type == "MatMul"]
    if len(ple_mm) != 1:
        raise ValueError(f"expected one MatMul (PLE projection) on the scaled embedding, found {len(ple_mm)}")
    ple_mm = ple_mm[0]
    weight = [x for x in ple_mm.input if x != scaled]
    if len(weight) != 1 or ple_mm.input[0] != scaled:
        raise ValueError("unexpected PLE projection inputs")
    ple_scale = [n for n in g.node if n.op_type == "Mul" and ple_mm.output[0] in n.input]
    if len(ple_scale) != 1:
        raise ValueError(f"expected one Mul after the PLE projection, found {len(ple_scale)}")
    ple_scale = ple_scale[0]
    scale_input = [x for x in ple_scale.input if x != ple_mm.output[0]]
    if len(scale_input) != 1:
        raise ValueError("unexpected PLE scale inputs")
    projected = ple_scale.output[0]

    g.input.append(helper.make_tensor_value_info("multimodal_embeds", TensorProto.FLOAT, ["batch", "sequence", 512]))
    g.input.append(helper.make_tensor_value_info("multimodal_mask", TensorProto.BOOL, ["batch", "sequence"]))
    g.initializer.append(helper.make_tensor("mm_unsqueeze_axes", TensorProto.INT64, [1], [-1]))
    mixed = "inputs_embeds_mixed"
    ple_mixed = "per_layer_projection_mixed"
    head = [
        helper.make_node("Unsqueeze", ["multimodal_mask", "mm_unsqueeze_axes"], ["mm_mask_3d"], name="mm_unsqueeze"),
        helper.make_node("Where", ["mm_mask_3d", "multimodal_embeds", scaled], [mixed], name="mm_where"),
    ]
    tail = [
        helper.make_node("MatMul", ["multimodal_embeds", weight[0]], ["mm_ple_linear"], name="mm_ple_linear"),
        helper.make_node("Mul", ["mm_ple_linear", scale_input[0]], ["mm_ple_scaled"], name="mm_ple_scale"),
        helper.make_node("Where", ["mm_mask_3d", "mm_ple_scaled", projected], [ple_mixed], name="mm_ple_where"),
    ]
    for n in g.node:
        if n is ple_mm:
            continue
        if n is not ple_scale:
            for i, x in enumerate(n.input):
                if x == projected:
                    n.input[i] = ple_mixed
        for i, x in enumerate(n.input):
            if x == scaled:
                n.input[i] = mixed
    index = list(g.node).index(muls[0]) + 1
    for k, n in enumerate(head):
        g.node.insert(index + k, n)
    index = list(g.node).index(ple_scale) + 1
    for k, n in enumerate(tail):
        g.node.insert(index + k, n)
    return {"mixed": mixed, "per_layer_projection_mixed": ple_mixed}
