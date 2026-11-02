"""PyTorch fp32 reference values for the Starlings.Models.Qwen3Embedding tests (transformers 5.18.0, CPU).

The tests compare the .NET embedder and reranker, run on a released graph, with these values; the token ids also check
the .NET tokenization without a graph. Every case runs alone, without padding.

- `embedding`: `AutoModel` `last_hidden_state` at the last token (the tokenizer appends `<|endoftext|>`), L2-normalized
  (the model card's transformers snippet). Records the token ids, the first 8 values and the sum of the embedding.
- `reranker`: `AutoModelForCausalLM` logits at the last position of `prefix + content + suffix` (each part tokenized
  without special tokens, the card's snippet); records the template's prefix and suffix ids, each case's content ids
  and the score `logit[yes = 9693] - logit[no = 2152]`.

Usage (environment `requirements.txt`):
    python make_dotnet_reference.py --model embedding --model-dir <Qwen/Qwen3-Embedding-0.6B@97b0c614> > embedding.json
    python make_dotnet_reference.py --model reranker --model-dir <Qwen/Qwen3-Reranker-0.6B@e61197ed> > reranker.json
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from qwen3_graph import NO_TOKEN_ID, YES_TOKEN_ID
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from verify_qwen3_onnx import CARD_DOCUMENTS, CARD_QUERIES, CARD_TASK, QUERY_PROMPT, RERANK_PREFIX, RERANK_SUFFIX

EMBEDDING_CASES = [
    ("query", CARD_QUERIES[0]),
    ("none", CARD_DOCUMENTS[0]),
    ("none", ""),
    ("none", "Ünïcödé 你好 <|im_start|> tabs\tand\nnewlines"),
]
RERANK_CASES = [(CARD_TASK, q, d) for q in CARD_QUERIES for d in CARD_DOCUMENTS] + [
    ("Judge relevance strictly", "Ünïcödé query 你好", "")
]


def embedding(model_dir: str) -> list[dict]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    out = []
    for prompt, text in EMBEDDING_CASES:
        ids = tok((QUERY_PROMPT if prompt == "query" else "") + text)["input_ids"]
        with torch.inference_mode():
            hidden = model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
        e = F.normalize(hidden, p=2, dim=0)
        head = [float(x) for x in e[:8]]
        out.append({"prompt": prompt, "text": text, "ids": ids, "head": head, "sum": float(e.sum())})
    return out


def reranker(model_dir: str) -> dict:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    prefix = tok.encode(RERANK_PREFIX, add_special_tokens=False)
    suffix = tok.encode(RERANK_SUFFIX, add_special_tokens=False)
    out = {"prefix_ids": prefix, "suffix_ids": suffix, "cases": []}
    for instruction, query, document in RERANK_CASES:
        text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        content = tok.encode(text, add_special_tokens=False)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([prefix + content + suffix])).logits[0, -1]
        score = float(logits[YES_TOKEN_ID] - logits[NO_TOKEN_ID])
        out["cases"].append(
            {"instruction": instruction, "query": query, "document": document, "content_ids": content, "score": score}
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["embedding", "reranker"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = embedding(args.model_dir) if args.model == "embedding" else reranker(args.model_dir)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
