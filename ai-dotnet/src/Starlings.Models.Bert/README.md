# BERT embeddings and reranking

Experimental CPU fp32 implementations for three explicitly pinned candidate contracts:
`sentence-transformers/all-MiniLM-L6-v2` (mean/L2, 256 tokens), `BAAI/bge-small-en-v1.5`
(CLS/L2, 512 tokens), and `cross-encoder/ms-marco-MiniLM-L6-v2` (paired inputs, raw scalar
logits, 512 tokens). Candidate parity is development evidence; protected release
qualification requires independent fixture adoption.

Load a reviewed candidate profile with `BertModelProfile.Load`. `BertTextEmbedder.Create`
provides embeddings; `BertReranker.Create` provides scores and document ranking. Both
verify the graph and tokenizer against the profile's immutable lock and load one reusable
ONNX session. Acquisition is explicit through `ModelAssetStore`; inference never downloads.
The application supplies the CPU native ONNX Runtime package.

BGE query instructions are explicit: call `Embed(text, "query")` to apply the declared
instruction; `Embed(text)` embeds text as supplied. Passages never acquire a query prefix.
Reranking returns logits with identity activation, not calibrated probabilities. Input pairs
are truncated with the pinned tokenizer's longest-first policy, including special tokens.

Batches are chunked and right-padded with mask 0. Threads default to two intra-op and one
inter-op worker. Calls are synchronous; cancellation follows ONNX Runtime graph-node
boundaries. Dispose an instance after outstanding calls complete. Streaming, training,
persistence, AOT, GPU providers, and unlisted model contracts are unsupported or untested.
