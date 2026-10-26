"""Check a released Qwen3 Embedding series graph against PyTorch fp32 (transformers 5.18.0), CPU.

Cases: the model card's example inputs, every text of the tokenizer corpus (`tokenizer-corpus-v1.jsonl`, including the
empty string), and two long inputs of about 4,000 and 8,000 tokens built by cycling through the corpus texts. Short
cases run through ONNX Runtime as one right-padded batch, long cases one at a time; the PyTorch reference runs every
case alone, without padding.

- `embedding`: reference `AutoModel` `last_hidden_state`, last-token pooling and L2 normalization (the card's
  transformers snippet). Records the largest difference of the hidden state over valid tokens and of the normalized
  embedding; checks that every input ends with `<|endoftext|>` (151643); computes the card's four query-document
  cosines from the graph's embeddings and from the card's left-padded batch, against the published
  [[0.7645568, 0.1414251], [0.1354974, 0.5999550]] (1e-3); cross-checks sentence-transformers `encode`.
- `reranker`: input `prefix + tokens("<Instruct>: {i}\\n<Query>: {q}\\n<Document>: {d}") + suffix` (the card's
  transformers snippet; the same text as the pinned `chat_template.jinja`, which is checked), reference
  `AutoModelForCausalLM` logits with the full language model head. Records the largest difference of the score
  (logit[yes = 9693] - logit[no = 2152] at the last position), of the two logits at every valid position (short
  cases) and of the hidden state; cross-checks sentence-transformers `CrossEncoder.predict`.

The numbers are evidence for the hosted repository's README, not acceptance bounds.

Usage:
    python verify_qwen3_onnx.py --model embedding|reranker --model-dir <snapshot> --graph <release>/model.onnx \\
        --corpus ../corpus/tokenizer-corpus-v1.jsonl --out verify.json [--threads 12]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import transformers
from qwen3_graph import EOS_TOKEN_ID, MAX_SEQUENCE, NO_TOKEN_ID, YES_TOKEN_ID

CARD_TASK = "Given a web search query, retrieve relevant passages that answer the query"
CARD_QUERIES = ["What is the capital of China?", "Explain gravity"]
CARD_DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
    "responsible for the movement of planets around the sun.",
]
CARD_COSINES = [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]
QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
LONG_TARGETS = (4000, 8000)


def corpus_texts(path: Path) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    return [(r["id"], r["text"]) for r in rows]


def long_text(tokenizer, texts: list[str], target: int) -> str:
    """Corpus texts joined by newlines, cycling, until the next text would pass `target` tokens."""
    parts: list[str] = []
    i = 0
    while True:
        candidate = "\n".join([*parts, texts[i % len(texts)]])
        if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) > target:
            return "\n".join(parts)
        parts.append(texts[i % len(texts)])
        i += 1


def session(path: str, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def run_ort(sess, batch: list[list[int]]) -> tuple[list[np.ndarray], np.ndarray]:
    width = max(len(ids) for ids in batch)
    ids = np.full((len(batch), width), EOS_TOKEN_ID, dtype=np.int64)
    mask = np.zeros((len(batch), width), dtype=np.int64)
    for row, seq in enumerate(batch):
        ids[row, : len(seq)] = seq
        mask[row, : len(seq)] = 1
    return sess.run(None, {"input_ids": ids, "attention_mask": mask}), mask


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def ort_outputs(sess, inputs: list[list[int]], long_ids: set[int]) -> list[list[np.ndarray]]:
    """Per case, the graph outputs over its valid tokens: short cases as one batch, long cases alone."""
    out: list[list[np.ndarray] | None] = [None] * len(inputs)
    short = [i for i in range(len(inputs)) if i not in long_ids]
    got, _ = run_ort(sess, [inputs[i] for i in short])
    for row, i in enumerate(short):
        out[i] = [g[row, : len(inputs[i])] for g in got]
    for i in sorted(long_ids):
        got, _ = run_ort(sess, [inputs[i]])
        out[i] = [g[0] for g in got]
        del got
        gc.collect()
    return out


def verify_embedding(args, cases: list[tuple[str, str]]) -> dict:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    card = [QUERY_PROMPT + q for q in CARD_QUERIES] + CARD_DOCUMENTS
    texts = [("card-query-0", card[0]), ("card-query-1", card[1]), ("card-doc-0", card[2]), ("card-doc-1", card[3])]
    texts += cases
    corpus = [t for _, t in cases if t]
    for target in LONG_TARGETS:
        texts.append((f"long-{target}", long_text(tokenizer, corpus, target - 1)))
    inputs = [tokenizer(t, truncation=True, max_length=MAX_SEQUENCE)["input_ids"] for _, t in texts]
    eos_last = all(ids[-1] == EOS_TOKEN_ID for ids in inputs)
    long_ids = {i for i, (name, _) in enumerate(texts) if name.startswith("long-")}

    sess = session(args.graph, args.threads)
    got = ort_outputs(sess, inputs, long_ids)
    del sess
    gc.collect()

    model = AutoModel.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    per_case = []
    ort_emb = []
    for (name, _), ids, (hidden,) in zip(texts, inputs, got, strict=True):
        with torch.inference_mode():
            ref = model(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy()
        e_ref, e_got = l2(ref[-1]), l2(hidden[-1])
        ort_emb.append(e_got)
        per_case.append(
            {
                "id": name,
                "tokens": len(ids),
                "hidden_max_abs": float(np.abs(hidden - ref).max()),
                "embedding_max_abs": float(np.abs(e_got - e_ref).max()),
                "cosine_to_reference": float(e_got @ e_ref),
            }
        )
        del ref
        gc.collect()

    # The card's snippet, verbatim: left-padded batch of the four texts, last position, L2.
    left = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    batch = left(card, padding=True, truncation=True, max_length=MAX_SEQUENCE, return_tensors="pt")
    with torch.inference_mode():
        snippet = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, -1], p=2, dim=1).numpy()
    del model
    gc.collect()
    cos_graph = (np.stack(ort_emb[:2]) @ np.stack(ort_emb[2:4]).T).tolist()
    cos_snippet = (snippet[:2] @ snippet[2:].T).tolist()

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(args.model_dir, device="cpu", model_kwargs={"dtype": torch.float32})
    st_q = st.encode(CARD_QUERIES, prompt_name="query", convert_to_numpy=True)
    st_d = st.encode(CARD_DOCUMENTS, convert_to_numpy=True)
    st_emb = np.concatenate([st_q, st_d])
    del st
    gc.collect()

    published = np.array(CARD_COSINES)
    return {
        "eos_last_all_inputs": eos_last,
        "cases": per_case,
        "hidden_max_abs": max(c["hidden_max_abs"] for c in per_case),
        "embedding_max_abs": max(c["embedding_max_abs"] for c in per_case),
        "card_cosines": {
            "published": CARD_COSINES,
            "graph": cos_graph,
            "graph_max_abs_vs_published": float(np.abs(np.array(cos_graph) - published).max()),
            "transformers_card_snippet": cos_snippet,
            "snippet_max_abs_vs_published": float(np.abs(np.array(cos_snippet) - published).max()),
            "within_1e-3": bool(np.abs(np.array(cos_graph) - published).max() <= 1e-3),
        },
        "sentence_transformers_embedding_max_abs": float(np.abs(st_emb - np.stack(ort_emb[:4])).max()),
        "card_snippet_embedding_max_abs": float(np.abs(snippet - np.stack(ort_emb[:4])).max()),
    }


def verify_reranker(args, cases: list[tuple[str, str]]) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prefix = tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)
    budget = MAX_SEQUENCE - len(prefix) - len(suffix)

    def content(query: str, doc: str) -> str:
        return f"<Instruct>: {CARD_TASK}\n<Query>: {query}\n<Document>: {doc}"

    pairs = [(f"card-{i}", CARD_QUERIES[i], CARD_DOCUMENTS[i]) for i in range(2)]
    pairs += [
        ("card-cross-0", CARD_QUERIES[0], CARD_DOCUMENTS[1]),
        ("card-cross-1", CARD_QUERIES[1], CARD_DOCUMENTS[0]),
    ]
    pairs += [(f"corpus-{name}", "What does this text say?", text) for name, text in cases]
    corpus = [t for _, t in cases if t]
    head = len(tokenizer(content(CARD_QUERIES[0], ""), add_special_tokens=False)["input_ids"])
    for target in LONG_TARGETS:
        doc_tokens = min(target, budget) - head - len(prefix) - len(suffix)
        pairs.append((f"long-{target}", CARD_QUERIES[0], long_text(tokenizer, corpus, doc_tokens)))
    inputs = []
    for _, q, d in pairs:
        ids = tokenizer(content(q, d), truncation=True, max_length=budget)["input_ids"]
        inputs.append(prefix + ids + suffix)
    long_ids = {i for i, (name, _, _) in enumerate(pairs) if name.startswith("long-")}

    template = (Path(args.model_dir) / "chat_template.jinja").read_text(encoding="utf-8")
    rendered = tokenizer.apply_chat_template(
        [{"role": "query", "content": CARD_QUERIES[0]}, {"role": "document", "content": CARD_DOCUMENTS[0]}],
        chat_template=template,
        tokenize=False,
    )
    template_matches = rendered == RERANK_PREFIX + content(CARD_QUERIES[0], CARD_DOCUMENTS[0]) + RERANK_SUFFIX

    sess = session(args.graph, args.threads)
    got = ort_outputs(sess, inputs, long_ids)
    del sess
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    per_case = []
    for i, ((name, _, _), ids, (yes_no, hidden)) in enumerate(zip(pairs, inputs, got, strict=True)):
        keep = 1 if i in long_ids else 0  # 0 keeps every position; both use the full 151,669-row head
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids]), logits_to_keep=keep, output_hidden_states=True)
        logits = out.logits[0][:, [NO_TOKEN_ID, YES_TOKEN_ID]].numpy()
        ref_hidden = out.hidden_states[-1][0].numpy()
        del out
        score_ref = float(logits[-1, 1] - logits[-1, 0])
        score_got = float(yes_no[-1, 1] - yes_no[-1, 0])
        entry = {
            "id": name,
            "tokens": len(ids),
            "score_reference": score_ref,
            "score_graph": score_got,
            "score_abs_diff": abs(score_got - score_ref),
            "hidden_max_abs": float(np.abs(hidden - ref_hidden).max()),
        }
        if keep == 0:
            entry["yes_no_logits_max_abs_all_positions"] = float(np.abs(yes_no - logits).max())
        per_case.append(entry)
        del ref_hidden, logits
        gc.collect()
    del model
    gc.collect()

    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(args.model_dir, device="cpu", model_kwargs={"dtype": torch.float32})
    st_scores = ce.predict([(q, d) for _, q, d in pairs[:4]], convert_to_numpy=True).tolist()
    del ce
    gc.collect()
    graph_scores = [c["score_graph"] for c in per_case[:4]]

    return {
        "chat_template_matches_prefix_content_suffix": template_matches,
        "cases": per_case,
        "score_max_abs": max(c["score_abs_diff"] for c in per_case),
        "yes_no_logits_max_abs": max(c.get("yes_no_logits_max_abs_all_positions", 0.0) for c in per_case),
        "hidden_max_abs": max(c["hidden_max_abs"] for c in per_case),
        "sentence_transformers": {
            "scores": st_scores,
            "graph_scores": graph_scores,
            "max_abs": float(np.abs(np.array(st_scores) - np.array(graph_scores)).max()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["embedding", "reranker"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    cases = corpus_texts(args.corpus)
    result = verify_embedding(args, cases) if args.model == "embedding" else verify_reranker(args, cases)
    result = {
        "model": args.model,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": __import__("sentence_transformers").__version__,
        "onnxruntime": ort.__version__,
        "attention": "sdpa",
        "seconds": round(time.time() - t0, 1),
    } | result
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
