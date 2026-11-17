# Guardrails against the L1B3RT4S collection, and the light level's remaining gaps (2026-11-16)

Follows [guardrails-2026-11-16](../guardrails-2026-11-16/README.md). Two things: a measurement of the guard levels on a real, public jailbreak collection, and four gaps of the light (pattern) level closed: random deletion and spliced letters, payloads split into variables, nested encodings, and attacks split across messages.

## The L1B3RT4S collection

- Source: [`elder-plinius/L1B3RT4S`](https://github.com/elder-plinius/L1B3RT4S) at commit `f441d65d901b29ca92ac20ee1afb5c9d0b6f8ef6` (2026-10-28), archive SHA-256 `3178b772fe9c76d6d91bc9aa3549529e5959945b3065900d4383d7ec00f8965f`, AGPL-3.0.
- `tools/fetch-l1b3rt4s.sh` downloads and verifies it into `.eval-data/` (ignored by Git) or `$STARLING_EVAL_DATA`. Nothing from it is committed, copied into tests or redistributed; the eval tool reads it at run time and prints only item ids (`file#section:hash`), sizes and scores, never text.
- Items: each vendor file (`ANTHROPIC.mkd`, `OPENAI.mkd`, `GROK-MEGA.mkd`, ...) is split on its model-name headings (`#` to `###` with text; lines of `#` alone are dividers and stay in the prompt); sections under 20 non-space characters are dropped; duplicates removed by SHA-256. Excluded: `SYSTEMPROMPTS.mkd` (leaked vendor system prompts, not attacks), `README.md`, `LICENSE`, the JSON token and shortcut lists. 135 prompts.
- Split: one file in three is held out, by the first byte of the SHA-256 of its name: 117 dev prompts (25 files), 18 held out (9 files). The author reuses templates across vendors, so held-out prompts resemble dev ones; and the light level's marker rules were written from public descriptions of this collection. Both make these numbers optimistic.
- Token bombs: `TOKEN80M8.mkd` (5.86 M variation selectors, 5.8 MB) and `TOKENADE.mkd` (0.47 M) are timed separately, not counted in the rates.
- Every item is an attack; there are no benign items from this source. False positives are measured on the sets of the first evidence record.

## Results

CPU (i7-5820K), Kev-0.8B fp32 under a 12 GB memory cap. "Before" is the light level of PR #335; "after" adds this change's views.

| Level | L1B3RT4S dev (117) | L1B3RT4S held out (18) | Latency p50 / p95 | Memory |
| --- | --- | --- | --- | --- |
| Light, obfuscation handling off | 68 | 14 | 0.3 / 3 ms | none |
| Light (PR #335) | 84 | 15 | 1.4 / 7 ms | none |
| Light (this change) | 84 | 15 | 1.9 / 9 ms | none |
| Kev-0.8B alone | 92 | 17 | 1.0 / 3.2 s | 3.6 GB |
| Balanced: light, then Kev-0.8B on what light does not flag | **104** | **17** | **1.2 ms** / 2.0 s | 3.6 GB |

No prompt was over Kev's limits (none not scanned). The obfuscation views account for all 15 `GROK-MEGA.mkd` prompts (attacks that read as such only once decoded) and the hidden-character file `#MOTHERLOAD.txt`. The light level's 33 dev misses are prompts without surface tricks or known markers: rewording, which the light level cannot follow. Kev catches most of them, and the cascade keeps the median at the light level's cost because light settles most prompts first.

Token bombs: `TOKEN80M8.mkd` is scanned in 157 ms and `TOKENADE.mkd` in 16 ms, both flagged `obfuscation` (score 1.0), nothing timed out; peak working set 177 MB.

## Gaps closed in the light level

- **Nested encodings.** Blobs (base64, hex) and smuggled payloads are decoded once more (depth 2), as are blobs inside a ROT13 copy: base64 of base64, base64 of ROT13, ROT13 of base64, hex of base64, tags of base64. A budget of 24 decoded texts and 256 K characters per check bounds the work; a text past the budget is cut, not skipped, so padding a payload does not hide its start.
- **Deletion, inserted letters, payloads split into variables, typos.** Phrases of 16 letters or more match the squashed view with up to 15% of their letters deleted, inserted or changed (`PhraseEditRatio`), found with Myers' bit-parallel approximate search (one pass, linear time). Shorter phrases match exactly.
- **Attacks split across messages.** The chat middleware also checks the last `ConversationWindow` user messages (default 4) joined, as one more input with source `GuardSource.Conversation` and message index -2, when the window holds two or more.

| Set | PR #335 | This change |
| --- | --- | --- |
| Generated obfuscation set (25 techniques, 75 rows; coverage, not recall) | 57/60 on the first 20 techniques | 75/75 (random deletion 3/3, insertion, variable split, three nested encodings) |
| 70 Unicode-heavy ordinary texts (false positives) | 3 | 3 (same texts) |
| Hand-written dev / held out | 19/24 and 0/24 / 2/16 and 0/16 | unchanged |

The insertion technique in the set puts a letter after about one character in nine; denser insertion is past the 15% tolerance and is not caught.

## Cost

| Input | PR #335 | This change |
| --- | --- | --- |
| Typical message | ~0.2 ms | ~0.3 ms |
| 50 KB of look-alike text | 34 / 42 ms | 37 / 50 ms |
| 84 KB real document | 47 / 76 ms | 66 / 97 ms |

The approximate phrase search adds about 20 ms per 84 KB (it runs on each squashed view). `PhraseEditRatio = 0` removes it; `DetectObfuscation = false` removes all views.

## Still not covered

- **Low-resource languages** (the Yong et al. translation attack) and **rewording**. Patterns cannot follow meaning; rules in languages that cannot be verified would do harm. The route is a model tier: Kev-0.8B (above), or an embedding tier on a multilingual embedder, which is a separate, measured proposal.
- Insertion denser than about 15% of a phrase, ciphers other than ROT13, encodings nested three deep, splits across more than `ConversationWindow` messages or across tool results.

## Reproduce

```sh
tools/fetch-l1b3rt4s.sh
dotnet build ai-dotnet/tools/Starlings.Guardrails.Eval.Tool -c Release
cd ai-dotnet/tools/Starlings.Guardrails.Eval.Tool/bin/Release/net10.0
dotnet Starlings.Guardrails.Eval.Tool.dll --set l1b3rt4s-dev                    # also l1b3rt4s-heldout, l1b3rt4s-bombs
dotnet Starlings.Guardrails.Eval.Tool.dll --set l1b3rt4s-dev --level balanced   # needs the cached Kev-0.8B bundle
dotnet Starlings.Guardrails.Eval.Tool.dll --set obfuscation
```
