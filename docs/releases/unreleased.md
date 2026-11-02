# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## BERT route: paraphrase-multilingual-MiniLM-L12-v2 draft contract (#182, unsupported)

- `BertModelProfile.Load` and `Parse` accept an eighth repository, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (XLM-R Unigram tokenizer with BERT segment ids, mean pooling, no L2, 128 tokens, 384 wide; candidate `model-pipeline/candidates/paraphrase-multilingual`). Text embedding through it is **unsupported**: the pinned `Starlings.Tokenizers` 0.3.4 refuses the model's `WhitespaceSplit` pre-tokenizer, so `BertTextEmbedder.Create` throws `UnsupportedTokenizerComponentException`. It is not a built-in profile (`BuiltInIds` still lists seven), has no registry manifest and claims no maturity. A test pins the refusal, so the Tokenizers bump that fixes it is noticed.
- `model-pipeline scaffold` takes `--token-type-ids` / `--no-token-type-ids` to state whether the graph takes `token_type_ids`; without either flag an existing entry keeps its value, and a value that contradicts the entry is refused.
