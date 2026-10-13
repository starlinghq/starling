"""GLiNER2 span-architecture entity path, restated as exportable modules and a pure-Python decoder (P-014).

Mirrors `gliner2.inference.runtime.ExtractorRuntimeMixin` at GLiNER2 source `55656fbf` for one entity schema
(`Schema().entities(labels)`, optional descriptions), `token_pooling` "first", `counting_layer` "count_lstm_v2":

    encoder(input_ids, attention_mask).last_hidden_state                       backbone (DeBERTa-v3-base)
    words  = hidden[text_word_first_positions]                                 first-subword pooling
    schema = hidden[schema_special_positions]   ([P], then one [E] per label)
    count_logits = count_pred(schema[0])         argmax 0 means "no entities" (gate)
    struct = count_embed(schema[1:], 1)[0]       GRU step 0 + DownscaledTransformer over the labels
    span_rep = SpanMarkerV0(words, spans)        [W, max_width, D]
    logits = einsum("wkd,md->mwk", span_rep, struct); scores = sigmoid(logits)

Only count index 0 is read for entities, and count rows are independent (the count axis is the
transformer's batch axis), so fixing the count at 1 inside the graph gives the same row 0 as
GLiNER2's `count_embed(schema[1:], pred_count)`. The gate (`pred_count == 0`) stays outside the graph.

`decode_entities` restates thresholding, span-to-character mapping, the default (span checkpoint)
confidence-first greedy overlap suppression, and the public formatting, so the C# port has a
Python statement of every discrete step to match.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch import nn


def span_indices(num_words: int, max_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """GLiNER2 `_compute_span_rep_core` indices for one padded length: (W*K, 2) safe spans and validity."""
    starts = torch.arange(num_words).unsqueeze(1).expand(-1, max_width)
    ends = starts + torch.arange(max_width).unsqueeze(0)
    valid = (ends < num_words).reshape(-1)
    safe = torch.stack(
        [
            torch.where(valid, starts.reshape(-1), torch.zeros_like(valid, dtype=torch.long)),
            torch.where(valid, ends.reshape(-1), torch.zeros_like(valid, dtype=torch.long)),
        ],
        dim=-1,
    )
    return safe, valid


def encoder_layer_explicit(x: torch.Tensor, layer: nn.TransformerEncoderLayer) -> torch.Tensor:
    """`nn.TransformerEncoderLayer` (post-norm, ReLU, batch_first, no mask, eval) with explicit operators.

    The statement a C# port follows, and the repair for the TorchScript exporter, which records the
    label count of the trace inside `nn.MultiheadAttention`'s reshapes.
    """
    attn = layer.self_attn
    d = x.shape[-1]
    h = attn.num_heads
    hd = d // h
    qkv = torch.nn.functional.linear(x, attn.in_proj_weight, attn.in_proj_bias)  # [B, M, 3D]
    q, k, v = qkv.chunk(3, dim=-1)
    b = x.shape[0]
    q = q.reshape(b, -1, h, hd).transpose(1, 2)  # [B, H, M, hd]
    k = k.reshape(b, -1, h, hd).transpose(1, 2)
    v = v.reshape(b, -1, h, hd).transpose(1, 2)
    weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (hd**0.5), dim=-1)
    ctx = torch.matmul(weights, v).transpose(1, 2).reshape(b, -1, d)
    x = layer.norm1(x + attn.out_proj(ctx))
    ff = layer.linear2(torch.relu(layer.linear1(x)))
    return layer.norm2(x + ff)


class EntityHeads(nn.Module):
    """Everything after the backbone: gathered word and schema vectors to count logits and span logits."""

    def __init__(self, extractor, explicit_attention: bool = False):
        super().__init__()
        self.max_width = extractor.max_width
        self.span_rep = extractor.span_rep.span_rep_layer  # SpanMarkerV0
        self.count_pred = extractor.count_pred
        ce = extractor.count_embed  # CountLSTMv2
        self.pos0 = nn.Parameter(ce.pos_embedding.weight[0].detach().clone(), requires_grad=False)
        self.gru = ce.gru
        self.transformer = ce.transformer  # DownscaledTransformer
        self.explicit_attention = explicit_attention

    def downscaled_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """DownscaledTransformer.forward, optionally with the explicit encoder layers."""
        t = self.transformer
        if not self.explicit_attention:
            return t(x)
        y = t.in_projector(x)
        for layer in t.transformer.layers:
            y = encoder_layer_explicit(y, layer)
        return t.out_projector(torch.cat([y, x], dim=-1))

    def struct_proj(self, labels: torch.Tensor) -> torch.Tensor:
        """count_embed(labels, 1)[0] for a batch: labels [B, M, D] -> [B, M, D]."""
        gru = self.gru
        gi = torch.nn.functional.linear(self.pos0, gru.weight_ih_l0, gru.bias_ih_l0)  # [3D]
        gh = torch.nn.functional.linear(labels, gru.weight_hh_l0, gru.bias_hh_l0)  # [B, M, 3D]
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * labels
        return self.downscaled_transformer(h + labels)

    def forward(self, words: torch.Tensor, schema: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """words [B, W, D], schema [B, 1+M, D] -> count_logits [B, 20], span_logits [B, M, W, K]."""
        count_logits = self.count_pred(schema[:, 0])
        b, w, d = words.shape
        safe, _ = span_indices(w, self.max_width)
        safe = safe.to(words.device).unsqueeze(0).expand(b, -1, -1)
        span_rep = self.span_rep(words, safe)  # [B, W, K, D]
        struct = self.struct_proj(schema[:, 1:])  # [B, M, D]
        span_logits = torch.einsum("bwkd,bmd->bmwk", span_rep, struct)
        return count_logits, span_logits


class EntityModel(nn.Module):
    """Route A: one graph from ids to logits. Inputs are GLiNER2's routing positions, padded per batch."""

    def __init__(self, extractor, explicit_attention: bool = False):
        super().__init__()
        self.encoder = extractor.encoder
        self.heads = EntityHeads(extractor, explicit_attention)

    def forward(self, input_ids, attention_mask, word_index, schema_index):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        d = hidden.shape[-1]
        words = torch.gather(hidden, 1, word_index.unsqueeze(-1).expand(-1, -1, d))
        schema = torch.gather(hidden, 1, schema_index.unsqueeze(-1).expand(-1, -1, d))
        count_logits, span_logits = self.heads(words, schema)
        return count_logits, span_logits


class Backbone(nn.Module):
    """Route B: DeBERTa-v3 last hidden state only."""

    def __init__(self, extractor):
        super().__init__()
        self.encoder = extractor.encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def sigmoid32(x: np.ndarray) -> np.ndarray:
    """torch.sigmoid in float32 (1 / (1 + exp(-x))), the score GLiNER2 thresholds."""
    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()


def decode_entities(
    *,
    count_logits: np.ndarray,
    span_logits: np.ndarray,
    labels: list[str],
    text: str,
    start_map: list[int],
    end_map: list[int],
    threshold: float,
    include_confidence: bool = True,
    include_spans: bool = True,
) -> dict[str, Any]:
    """Public GLiNER2 entity output for one sample from the graph outputs (default overlap policy).

    `span_logits` is [M, W, K] for M labels in schema order (duplicates included), W >= text words.
    `text` is the normalized text. Returns what `GLiNER2.extract(..., format_results=True)` returns.
    """
    n = len(start_map)
    if int(np.argmax(count_logits)) <= 0:
        return {"entities": {}}
    scores = sigmoid32(span_logits)[:, :n, :]
    thr = np.float32(threshold)  # torch compares an fp32 tensor with a Python float in fp32
    out: OrderedDict[str, Any] = OrderedDict()
    for name in OrderedDict.fromkeys(labels):  # entity_order: dict keys, first occurrence
        idx = labels.index(name)
        s = scores[idx]
        spans = []
        starts, widths = np.nonzero(s >= thr)  # row-major, as torch.where
        for start, width in zip(starts.tolist(), widths.tolist(), strict=True):
            end = start + width + 1
            if end > n:
                continue
            cs, ce = start_map[start], end_map[end - 1]
            surface = text[cs:ce].strip()
            if surface:
                spans.append((surface, float(s[start, width]), cs, ce))
        ranked = sorted(spans, key=lambda sp: sp[1], reverse=True)
        kept = []
        for cand in ranked:
            if any(cand[2] < e[3] and e[2] < cand[3] for e in kept):
                continue
            kept.append(cand)
        out[name] = kept
    return {"entities": _format_entities(out, include_confidence, include_spans)}


def _format_entities(spans_by_label, include_confidence: bool, include_spans: bool) -> dict:
    """`_format_spans` followed by `format_entity_dict` (dedup on lower-cased text, plus offsets if kept)."""
    formatted = {}
    for name, spans in spans_by_label.items():
        unique, seen = [], set()
        for text, conf, start, end in spans:
            if include_spans:
                item = {"text": text, "start": start, "end": end}
                if include_confidence:
                    item = {"text": text, "confidence": conf, "start": start, "end": end}
                key = (text.lower(), start, end)
            else:
                item = {"text": text, "confidence": conf} if include_confidence else text
                key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        formatted[name] = unique
    return formatted


def threshold_margins(span_logits: np.ndarray, n: int, threshold: float, max_width: int) -> dict[str, Any]:
    """Valid candidates closest to the threshold (score units), for boundary-flip accounting."""
    scores = sigmoid32(span_logits)[:, :n, :]
    m, w, k = scores.shape
    valid = (np.arange(w)[:, None] + np.arange(k)[None, :]) < n
    vals = scores[:, valid]
    if vals.size == 0:
        return {"valid_candidates": 0}
    margin = np.abs(vals.astype(np.float64) - np.float64(np.float32(threshold)))
    return {
        "valid_candidates": int(vals.size),
        "above_threshold": int((vals >= np.float32(threshold)).sum()),
        "min_margin": float(margin.min()),
        "within_1e-3": int((margin < 1e-3).sum()),
        "within_1e-2": int((margin < 1e-2).sum()),
        "saturated_at_1": int((vals == np.float32(1.0)).sum()),
    }
