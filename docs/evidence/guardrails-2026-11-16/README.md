# Guardrails: first measurement (2026-11-16)

What the experimental guardrails (`Starlings.Guardrails`, `Starlings.Guardrails.Extensions.AI`, `Starlings.Models.Kev.Guardrails`) detect on input text, and what each level costs. Input side only: prompt injection, jailbreak and prompt-extraction attempts against an assistant. The output (harmful response) questions are **unmeasured**: no licensed evaluation set has been chosen.

## Setup

- Linux x64, Intel i7-5820K (6 cores, AVX2), CPU only, ONNX Runtime 1.30.0; Kev-0.8B bundle `Starlinghq/kev-0.8b-onnx` at `8a9fe017` (fp32), Kev's default `KevOptions` (calibrated temperature).
- `tools/Starlings.Guardrails.Eval.Tool`, one process per row, each model run under `systemd-run --user --scope -p MemoryMax=12G`.
- Data, hand-written for this measurement, no third-party text: `data/dev.jsonl` (24 attacks, 24 benign look-alikes) for choosing rules, questions and thresholds; `data/heldout.jsonl` (16 attacks, 16 benign look-alikes) written after the dev runs and before the pattern rules were ported, then run once per level. Benign items are deliberately close to attacks: "ignore" and "instructions" in ordinary requests, questions about jailbreaks and prompt injection, role-play requests, and the same carrier formats (reviews, search results, invites, PDF footers, code comments) without an embedded instruction.
- Caveats: one author wrote the rules, the questions and both sets; 32 held-out items give wide intervals (one item moves TPR or FPR by about 6 points); no multilingual or long-document cases beyond a few short French, German and Spanish items.

## Held-out results (reported)

| Level | Guard | AUC | Attacks caught (TPR) | Benign flagged (FPR) | Latency p50 / p95 | Peak working set |
| --- | --- | --- | --- | --- | --- | --- |
| Light | `patterns` | 0.56 | 2/16 (0.12) | 0/16 (0.00) | 0.02 / 3.5 ms | 40 MB |
| (model alone) | `kev-0.8b` | 0.84 | 12/16 (0.75) | 3/16 (0.19) | 463 / 546 ms | 2,267 MB |
| Balanced | `patterns>kev-0.8b` | 0.83 | 12/16 (0.75) | 3/16 (0.19) | 478 / 568 ms | 2,259 MB |
| Strict | `patterns>kev-0.8b` | 0.84 | 12/16 (0.75) | 3/16 (0.19) | 474 / 543 ms | 2,266 MB |

After the held-out runs, one pattern rule was narrowed (`DAN` made case-sensitive, so the name "Dan" does not flag); the light level was re-run on both sets and its counts are identical (dev 19/24 and 0/24, held out 2/16 and 0/16).

Kev's misses: instructions embedded in carried content (a product review, a search result, a calendar invite) and one context-window dump request. Kev's false positives: a question about jailbreak techniques for testing, IKEA assembly instructions, and a "hypothetically" mortgage question. The model load takes about 10 s and is not in the latency.

## Dev results (used for choices, not a quality claim)

| Guard | AUC | TPR | FPR | p50 |
| --- | --- | --- | --- | --- |
| `patterns` | 0.90 | 19/24 | 0/24 | 0.02 ms |
| `kev-0.8b`, choice question (default) | 0.87 | 21/24 | 5/24 | 438 ms |
| `kev-0.8b`, one noul question | 0.91 | 23/24 | 13/24 | 370 ms |

The noul question ranks slightly better but its scores sit high: at the 0.5 threshold it flags over half of the benign items, so the choice question is the default. The pattern rules were written while reading the dev set: their drop from 19/24 on dev to 2/16 on held-out is the overfitting this split exists to expose.

An earlier spike on the dev set also tried multilingual-e5-small with 8 exemplars per side (AUC 0.88, 6 ms, about 680 MB) and three noul questions per category on Kev (FPR 0.67 at 0.5); neither is in the packages.

## Conclusions

- The patterns are a free first tier with no false positives here but low recall on unseen phrasings. They are not a guard on their own.
- Zero-shot Kev-0.8B is a weak detector: 3 in 4 attacks at 1 in 5 false positives, half a second per text on CPU, 2.3 GB. It misses indirect injection, the case that matters most for tool results.
- Kev-4B was not run (16.9 GB fp32; same training, so no large change is expected, unmeasured).
- A "be very sure" level needs a purpose-trained guard model (a model admission); the cascade and the `IContentGuard` contract are built to take one.

## Obfuscation handling in the light level (added the same day)

The pattern guard gained views that undo the character tricks used to slip attacks past guardrails, multilingual rules, and markers of published jailbreak kits. Sources:

- Hackett et al., "Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails" ([arXiv 2504.11168](https://arxiv.org/abs/2504.11168)): 12 character-injection techniques against six commercial and open guardrails. Emoji smuggling (bytes hidden in variation selectors) bypassed them 100% of the time, Unicode tag smuggling 90% / 82% (injection / jailbreak), leetspeak digits 81% / 95%, bidirectional text 79% / 99%, upside-down text 64% / 100%.
- Unicode tag "ASCII smuggling" (U+E0000..E007F) and variation-selector smuggling ([garak reference](https://reference.garak.ai/en/latest/ascii_smuggling.html)); NFKC normalization leaves both in place, and stripping them as noise deletes the payload unread, so the guard decodes them first.
- Yong et al., "Low-Resource Languages Jailbreak GPT-4" ([arXiv 2310.02446](https://arxiv.org/abs/2310.02446)): translation into low-resource languages raised the bypass rate from under 1% to 79%.
- The L1B3RT4S jailbreak collection (GitHub `elder-plinius/L1B3RT4S`, AGPL-3.0, read as data only; no text copied): override banners (`[DISREGARD PREV. INSTRUCTS]`, `<NEW_PARADIGM>`, `{*CLEAR YOUR MIND*}`), dividers that spell LOVE PLINY between separators, often carrying invisible tag characters, "GODMODE", refusal inversion, and requests to answer in leetspeak. Only markers the sources confirmed became rules.

### Data

Both files are written by the committed generator `data/make-sets.py` with ASCII escapes, so hidden characters show in review.

- `benign-unicode.jsonl`, 70 hand-written ordinary texts, written **before** the new rules: emoji sequences (ZWJ families, keycaps, skin tones), subdivision flags (which use tag characters legitimately), ideographic variation sequences, Arabic and Hebrew with right-to-left marks, Persian and Hindi with joiners, i18n isolates, a byte-order mark, soft hyphens, gamer leetspeak, JWT, data-URI and base64 decode requests, hex hashes, zalgo, upside-down, full-width and math-bold text, spaced emphasis, "god mode in Doom", "I love Pliny the Elder", "never refuse a dessert", "answer in pig latin", and everyday "ignore my last message" sentences in each language with a rule. **Its false-positive rate is the quality number.**
- `obfuscation.jsonl`, 60 rows generated from three attack phrases that the plain rules catch, each put through the paper's 12 techniques and 8 more (base64, hex, ROT13, reversed, emoji between letters, math-bold, random capitals, leet with spaces). The guard inverts these exact transforms, so this measures **coverage of techniques, not real-world recall**.

### Results

| Set | Obfuscation handling off | On |
| --- | --- | --- |
| `obfuscation`: attacks caught | 12/60 | 57/60 |
| `benign-unicode`: ordinary texts flagged | 3/70 | 3/70 |
| `dev` (24 + 24) | 19/24 caught, 0/24 flagged | unchanged |
| `heldout` (16 + 16) | 2/16 caught, 0/16 flagged | unchanged |

Per technique with handling on, every row of every technique is caught except random deletion (0/3; the paper also found it the least effective attack). Emoji smuggling, tag smuggling, bidi, base64, hex, ROT13, reversed, upside-down, leet, homoglyphs, diacritics, underline, spacing, zero-width and emoji-between-letters also flag `obfuscation` (the attack shows only once decoded); full-width, math-bold and random capitals do not, because NFKC and case-insensitive matching already undo them in the plain view.

The 3 benign texts flagged are the mention-versus-use limit of patterns, the same with handling off: "Never refuse a dessert" (two jailbreak signals), a question about the `<|endoftext|>` token, and a quote of "ignore previous instructions" in a security-class question. The obfuscation views add no false positive on this set. One rule was fixed after the set was run: the Russian and Ukrainian rules matched the stem `правил`, which also starts `правильный` ("correct"); they now require `правила`/`правило`. Held-out recall does not move: its misses are paraphrases, not obfuscation, which is the model tier's job.

### Cost

CPU (i7-5820K), one thread, p50 / p95 over 20 runs after a warm-up:

| Input | Handling off | Handling on |
| --- | --- | --- |
| Typical message (sets above) | ~0.04 ms | ~0.2 ms |
| 1 KB of look-alike text | 0.4 / 0.8 ms | 1.8 / 6 ms |
| 50 KB of look-alike text (worst case: trigger words everywhere) | 13 / 19 ms | 34 / 42 ms |
| 84 KB real document (`docs/plan/fast-faithful-ai-dotnet.md`) | 21 / 28 ms | 47 / 76 ms |

Memory stays at the 40 to 44 MB of the process; nothing is loaded. The cost of long text is about 0.3 ms per rule pass per 50 KB, and AI-related prose contains the rules' keywords, so the keyword prefilter saves little there. Knobs: `DetectObfuscation = false`, a lower `MaxDecodeChars` (the whole-text ROT13, reversed and upside-down copies; 4,096 by default), or not checking tool results in the middleware. Profiling found two rules 10x slower than the rest (case-insensitive matching on Turkish and Cyrillic letters loses .NET's vectorized search); they now spell case variants out.

### Not covered

- Low-resource languages (the Yong et al. attack): no rules, because a wrong rule is worse than none; the model tier is the route. Mixed-language "sandwich" prompts likewise.
- Paraphrased and novel attacks, random character deletion, ciphers other than ROT13, Morse, payloads split across messages, and decoding more than one level deep (base64 of base64).
- Attacks after the first 4,096 characters, for the whole-text ROT13, reversed and upside-down copies only.

## Reproduce

```sh
dotnet build ai-dotnet/tools/Starlings.Guardrails.Eval.Tool -c Release
cd ai-dotnet/tools/Starlings.Guardrails.Eval.Tool/bin/Release/net10.0
dotnet Starlings.Guardrails.Eval.Tool.dll --set heldout --level light
dotnet Starlings.Guardrails.Eval.Tool.dll --set heldout --level model      # also balanced, strict; needs the cached Kev-0.8B bundle
dotnet Starlings.Guardrails.Eval.Tool.dll --set dev --level model --question noul
dotnet Starlings.Guardrails.Eval.Tool.dll --set obfuscation                # per technique; --obfuscation off for the baseline
dotnet Starlings.Guardrails.Eval.Tool.dll --set benign-unicode
dotnet Starlings.Guardrails.Eval.Tool.dll --set dev --doc-kb 50            # or --doc-file <path>
python3 -I ../../../data/make-sets.py                                      # regenerates benign-unicode.jsonl and obfuscation.jsonl
```
