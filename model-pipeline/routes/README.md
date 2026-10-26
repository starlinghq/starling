# Route contract tables

A route contract table is the allow-list of an onboarding route as data: one entry per upstream repository the route
implements, with the processing contract its code must reproduce. Both implementations of the route read the same
file, so they cannot drift apart.

| File | Route | Read by |
| --- | --- | --- |
| [`bert-onnx-cpu-v1/contracts.json`](bert-onnx-cpu-v1/contracts.json) | `bert-onnx-cpu-v1` | `src/model_pipeline/onboarding_bert.py` (`load_contracts`, `validate_contracts`) and `ai-dotnet/src/Starlings.Models.Bert` (embedded resource `Starlings.Models.Bert.Contracts.json`; `BertModelProfile.SupportedRepositories`) |

The other routes (e5, Qwen, Kev, Whisper) still keep their allow-lists in code.

## `bert-onnx-cpu-v1/contracts.json`

`format` is `starlings.bert-route-contracts`, `format_version` is `1` and `route` is `bert-onnx-cpu-v1`. `contracts` maps
each Hugging Face repository (`<owner>/<name>`) to an object with exactly these fields:

| Field | Meaning | Allowed values |
| --- | --- | --- |
| `task` | Typed API the profile serves | `embeddings` or `reranking` |
| `max_tokens` | Token limit, special tokens and prompt included | positive integer |
| `pooling` | Pooling of the graph output | `mean` or `cls` (embeddings); `none` (reranking) |
| `hidden_size` | Width of the pooled output | positive integer |
| `normalize` | Vector normalization | `l2` or `none` (embeddings); `none` (reranking) |
| `vocabulary_size` | Exclusive upper bound of token ids | positive integer |
| `cls_token_id`, `sep_token_id`, `pad_token_id` | First, separator and right-padding ids | inside the vocabulary |
| `output_name` | Graph output to pool | non-empty string |
| `tokenizer_profile` | Owned tokenizer pipeline | `bert-wordpiece-uncased` or `xlmr-unigram` |
| `token_type_ids` | Whether the graph takes segment ids | boolean |
| `prompts` | Named prefixes | object of strings that includes `"none": ""` |
| `score_activation_source` (optional, reranking only) | Where the reference reads the score activation | `config` (default) or `explicit` |

Both readers refuse a table with another format, a duplicate key, a missing or unknown field, a value of the wrong
type or any value outside the column above, and the .NET package fails on first use rather than serve a profile from
a malformed table. A candidate's `behavior` must then equal its entry field for field (`validate_candidate_behavior`
and `BertModelProfile.Parse`).

Add an entry with `model-pipeline scaffold <candidate>` ([ONBOARDING.md](../ONBOARDING.md#scaffold-a-bert-route-candidate)),
not by hand. An entry is not a support claim: it lets the route run the candidate; gates, maturity and protected
evidence are unchanged.
