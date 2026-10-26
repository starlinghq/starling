"""Exportable Qwen3 Embedding series graphs (transformers 5.18.0, `Qwen3Model`), used non-autoregressively.

Both checkpoints are a Qwen3 decoder (hidden 1024, 28 layers, vocabulary 151,669, tied embeddings). The graphs take
`input_ids` and `attention_mask` (int64, [batch, sequence], right padding) and run the full causal stack once: no
position_ids input (positions are 0..sequence-1, as transformers uses without a cache) and no KV cache.

`Qwen3Graph` builds the attention mask explicitly instead of through transformers' mask utilities: an additive float32
mask (0 or the float32 minimum, never boolean) that allows key k for query q when k <= q and attention_mask[k] is 1.
With right padding key 0 is valid for every query row, so no row is fully masked. The mask is passed to `Qwen3Model` as
the precomputed `{"full_attention": mask}` mapping (the model has no sliding-window layers).

- `embedding`: output `last_hidden_state` ([batch, sequence, 1024], after the final norm). Last-token pooling (the
  hidden state at sum(attention_mask) - 1) and L2 normalization stay outside the graph.
- `reranker`: outputs `yes_no_logits` ([batch, sequence, 2]) and `last_hidden_state`. `yes_no_logits` is the language
  model head restricted to two rows of the tied embedding matrix, in the order [no = 2152, yes = 9693]; the rows are
  copied into a 2x1024 buffer, so the graph neither duplicates the embedding matrix nor computes 151,669-wide logits.
  The reranker score is yes - no at the last valid position (sentence-transformers `LogitScore`).
"""

from __future__ import annotations

import torch
from torch import nn

NO_TOKEN_ID = 2152
YES_TOKEN_ID = 9693
HEAD_ROWS = (NO_TOKEN_ID, YES_TOKEN_ID)
EOS_TOKEN_ID = 151643  # <|endoftext|>: the embedding tokenizer appends it, and both use it as the padding id
MAX_SEQUENCE = 8192


class Qwen3Graph(nn.Module):
    def __init__(self, model: nn.Module, kind: str):
        super().__init__()
        if kind not in ("embedding", "reranker"):
            raise ValueError(kind)
        self.m = model
        self.kind = kind
        if kind == "reranker":
            rows = model.embed_tokens.weight.detach()[list(HEAD_ROWS)].clone().contiguous()
            self.register_buffer("head", rows, persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embeds = self.m.embed_tokens(input_ids)
        seq = input_ids.shape[1]
        positions = torch.arange(seq, device=input_ids.device)
        causal = positions[None, :] <= positions[:, None]  # [T(query), T(key)]
        key_valid = attention_mask.to(torch.bool)[:, None, None, :]  # [B, 1, 1, T]
        zero = torch.zeros((), dtype=embeds.dtype, device=input_ids.device)
        low = torch.full((), torch.finfo(embeds.dtype).min, dtype=embeds.dtype, device=input_ids.device)
        mask = torch.where(key_valid & causal[None, None, :, :], zero, low)  # [B, 1, T, T]
        out = self.m(
            inputs_embeds=embeds,
            attention_mask={"full_attention": mask},
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        )
        hidden = out.last_hidden_state
        if self.kind == "embedding":
            return hidden
        return torch.matmul(hidden, self.head.t()), hidden


def output_names(kind: str) -> list[str]:
    return ["last_hidden_state"] if kind == "embedding" else ["yes_no_logits", "last_hidden_state"]


def load_backbone(model_dir: str, attn: str = "sdpa") -> nn.Module:
    """`Qwen3Model` (no language model head), float32, as sentence-transformers' Transformer module loads it."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32, attn_implementation=attn)
    model.eval()
    return model
