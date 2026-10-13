"""Exportable text path of EmbeddingGemma 2 (transformers 5.19.0, `EmbeddingGemma2TextModel`).

The text embedding is: token embedding (scaled by sqrt(512)), per-layer embedding (PLE) projection,
24 bidirectional encoder layers (sliding window on 20 layers, full attention on layers 5, 11, 17,
23), final RMSNorm, then `embedding_projection` (512 to 768, no bias) on every token. The
sentence-transformers modules mean-pool those 768-wide token embeddings over the attention mask
(prompt tokens included) and L2-normalize. Projection and mean pooling commute because the
projection is linear and has no bias.

`TextPath` restates `EmbeddingGemma2TextModel.forward` with the masks built explicitly from the
2-D attention mask, so the graph has two inputs (`input_ids`, `attention_mask`, int64) and two
outputs: `last_hidden_state` (512, after the final norm, before the projection; diagnostic) and
`token_embeddings` (768, the sentence-transformers module output). Pooling and normalization
stay outside the graph, as for e5.

Masks are additive float32 (0 or the float32 minimum), never boolean: a padded query row whose
sliding window holds no valid key would give NaN with a boolean mask, and NaN times a zero
pooling weight is still NaN.

The sliding-window rule is transformers' `sliding_window_bidirectional_overlay`:
`abs(q - k) <= sliding_window` with `sliding_window` = 512, so a token sees up to 1,025 positions
(the card's "1024 tokens"). Sequences longer than 513 tokens exercise it.
"""

from __future__ import annotations

import torch
from torch import nn


class TextPath(nn.Module):
    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.m = text_model
        self.window = int(text_model.config.sliding_window)
        self.layer_types = list(text_model.config.layer_types)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        m = self.m
        inputs_embeds = m.embed_tokens(input_ids)
        per_layer_inputs = m.ple(inputs_embeds)
        seq = input_ids.shape[1]
        positions = torch.arange(seq, device=input_ids.device)
        position_ids = positions.unsqueeze(0)

        neg = torch.finfo(inputs_embeds.dtype).min
        key_valid = attention_mask.to(torch.bool)[:, None, None, :]  # [B, 1, 1, T]
        near = (positions[:, None] - positions[None, :]).abs() <= self.window  # [T, T]
        zero = torch.zeros((), dtype=inputs_embeds.dtype, device=input_ids.device)
        low = torch.full((), neg, dtype=inputs_embeds.dtype, device=input_ids.device)
        full = torch.where(key_valid, zero, low)  # [B, 1, 1, T], broadcast over queries
        sliding = torch.where(key_valid & near[None, None, :, :], zero, low)  # [B, 1, T, T]
        masks = {"full_attention": full.expand(-1, -1, seq, -1), "sliding_attention": sliding}

        hidden = inputs_embeds
        rope = {t: m.rotary_emb(hidden, position_ids, t) for t in sorted(set(self.layer_types))}
        for i, layer in enumerate(m.layers):
            t = self.layer_types[i]
            hidden = layer(hidden, per_layer_inputs[:, :, i, :], attention_mask=masks[t], position_embeddings=rope[t])
        hidden = m.norm(hidden)
        return hidden, m.embedding_projection(hidden)


def load_text_model(model_dir: str, attn: str = "sdpa") -> nn.Module:
    """The text backbone alone, fp32, as sentence-transformers' Transformer module wraps it (language_model)."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_dir,
        dtype=torch.float32,
        attn_implementation=attn,
        vision_config=None,
        audio_config=None,
    )
    model.eval()
    return model.language_model


def pool(token_embeddings, mask):
    """sentence-transformers Pooling (mean over the mask) and Normalize, in numpy float32."""
    import numpy as np

    m = mask[..., None].astype(np.float32)
    summed = (token_embeddings * m).sum(axis=1)
    mean = summed / np.clip(m.sum(axis=1), 1e-9, None)
    return mean / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)
