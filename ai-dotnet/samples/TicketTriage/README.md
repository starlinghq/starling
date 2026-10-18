# TicketTriage sample

A help desk has labelled history in English only, but tickets also arrive in German, Spanish, French and Japanese. This sample trains a ticket router on the English tickets and uses it on all of them. Everything runs locally in an ordinary ML.NET pipeline: multilingual-e5-small embeddings (`Starlings.Models.E5.MLNet`) are the features for an ML.NET trainer, and GLiNER2 (`Starlings.Models.Gliner2`) extracts order numbers, products and dates. It references published packages only and requires the .NET 10 SDK.

From this directory:

```sh
dotnet build -c Release
dotnet run -c Release --no-build -- download
dotnet run -c Release --no-build -- train
dotnet run -c Release --no-build -- triage
dotnet run -c Release --no-build -- triage "Mon colis est arrivé ouvert et il manque un article."
```

`download` acquires and verifies the e5 (about 470 MB) and GLiNER2 (about 800 MB) assets. It is the only command that uses the network. `train` and `triage` load verified local files and fail rather than download. All commands accept `--cache <dir>`; otherwise they use `$STARLING_MODELS` or the user cache (`~/.cache/starlings/models` on Linux). `train` and `triage` accept `--model <zip>` (default `triage-model.zip` in the current directory). `StarlingsVersion` pins the published packages, and `packages.lock.json` records the dependency graph; verify it with `dotnet restore --locked-mode`.

## The pipeline

`data/train.tsv` holds 80 English tickets, 16 in each of five queues: billing, shipping, technical, account and returns. `data/test.tsv` holds 50 held-out tickets, 10 each in English, German, Spanish, French and Japanese. Neither file is a translation of the other.

`train` builds this pipeline:

```text
Text ─► ApplyE5Embedding ─► Cache ─► MapValueToKey(Queue) ─► Concatenate(Features) ─► SdcaMaximumEntropy ─► MapKeyToValue
        "query: " prefix,            └──────────────────────── saved to triage-model.zip ─────────────────────────────┘
        384 floats per row
```

- **The embedding is a pretrained transform.** `Fit` loads and verifies the model and trains nothing. The `query: ` prefix is e5's prefix for symmetric tasks such as classification and ticket-to-ticket similarity.
- **Cache after the embedding.** SDCA makes many passes over its input, and the embedding column is computed lazily on every pass. `ml.Data.Cache` runs e5 once per ticket instead of once per ticket per epoch.
- **A baseline.** The same label mapping and trainer run over ML.NET's own `FeaturizeText` bag of word and character n-grams, so the effect of the embeddings is visible. Both trainers are seeded and single-threaded, so the numbers below repeat run to run.
- **Saving.** The e5 transformer cannot be saved, because its model is the pinned asset lock, resolved at run time; `Model.Save` throws for it. So `train` saves only the classifier chain, with the schema it consumes (the input columns plus `Embedding`). `triage` loads it and puts a fresh e5 transformer in front of it, with the same column names and prefix.

`triage` composes one `TransformerChain` and one `PredictionEngine`:

```text
e5 embedding (re-attached) ─► saved classifier ─► CustomMapping(GLiNER2: order number, product, date)
```

The `CustomMapping` shows that any local model can join a pipeline as a row-wise transform. It is not saved either. For each ticket, `triage` prints:

- the predicted queue and the runner-up, with their probabilities;
- a request for human review when the two likeliest queues are within 15 points of each other;
- the extracted entities;
- the most similar past ticket, by cosine similarity of the same embeddings, as a link an agent could follow.

## A real run

Linux x64 CPU, .NET SDK 10.0.401 (runtime 10.0.12), Starlings 0.2.0, on 2026-10-17, from an existing cache passed with `--cache ~/.cache/starling-models`. `train` output:

```text
Trained on 80 English tickets (embedding and training 5.6 s).

Accuracy on held-out tickets (the training set is English only)
language  tickets  bag-of-n-grams  e5 embeddings
en                10     70 %          100 %
de                10     50 %          100 %
es                10     50 %          100 %
fr                10     30 %          100 %
ja                10     20 %          100 %
all non-English   40     38 %          100 %

e5 pipeline on non-English tickets: macro accuracy 1.000, log-loss 0.246

Confusion table
             ||========================================
PREDICTED    ||     0 |     1 |     2 |     3 |     4 | Recall
TRUTH        ||========================================
0.   billing ||     8 |     0 |     0 |     0 |     0 | 1.0000
1.  shipping ||     0 |     8 |     0 |     0 |     0 | 1.0000
2. technical ||     0 |     0 |     8 |     0 |     0 | 1.0000
3.   account ||     0 |     0 |     0 |     8 |     0 | 1.0000
4.   returns ||     0 |     0 |     0 |     0 |     8 | 1.0000
             ||========================================
Precision    ||1.0000 |1.0000 |1.0000 |1.0000 |1.0000 |

Saved classifier: <sample directory>/triage-model.zip
```

`triage` output, with no arguments, on its built-in tickets:

```text
ticket:   Hallo, meine Bestellung 40917 sollte am 12. Oktober ankommen, ist aber immer noch nicht da.
queue:    shipping 40 %, then billing 37 %  -> needs human review
entities: order number: 40917; date: 12. Oktober
similar:  [shipping] I need to change the delivery address for order 77140 before it ships. (cosine 0.846)

ticket:   Bonjour, le grille-pain TurboToast 3000 ne chauffe plus, je voudrais le renvoyer.
queue:    returns 39 %, then technical 33 %  -> needs human review
entities: product: TurboToast 3000
similar:  [returns] The blender I received is defective, I want a replacement. (cosine 0.880)

ticket:   ログインしようとするとパスワードが違うと言われます。
queue:    account 97 %, then technical 1 %
entities: (none)
similar:  [account] I forgot my password and the reset email never arrives. (cosine 0.851)

ticket:   My SmartHub 2 stopped responding after the firmware update on October 3.
queue:    technical 92 %, then billing 4 %
entities: product: SmartHub 2; date: October 3
similar:  [technical] I cannot connect the SmartHub 2 to my Wi-Fi network. (cosine 0.924)

ticket:   Hi, quick question about my account and an order.
queue:    account 43 %, then shipping 28 %  -> needs human review
entities: (none)
similar:  [account] Please delete my account and all my personal data. (cosine 0.895)
```

## Reading the results

The bag of n-grams knows only the words it saw in English training text. It does reasonably on English, then falls toward chance (20% for five queues) as the vocabulary moves away from English; Japanese shares almost no n-grams with the training data. The e5 embeddings put tickets with the same meaning close together whatever the language, so a classifier trained on English vectors transfers.

Do not over-read the 100%: the test tickets are short, hand-written and clearly in one queue. On real tickets, expect lower accuracy and more requests for review. The review rule is doing its job above. "Ne chauffe plus, je voudrais le renvoyer" is genuinely between technical and returns, and the vague last ticket is between account and shipping. The German shipping ticket was routed correctly but with a narrow margin; with 80 training tickets the probabilities are soft. More labelled history is the fix, not a lower threshold.

GLiNER2 found the order number and the German date, and nothing in the Japanese ticket, which contains no order number, product or date. It is not infallible: for "Mon colis est arrivé ouvert et il manque un article." (routed to shipping, 56%, then returns 27%) it reported `product: colis`, the French word for parcel. Training time includes embedding the 80 tickets; `triage` takes about 20 s here, mostly loading the two models.

## Limits

This is a small-data demonstration, not a tuned router. `triage` re-embeds the 80 past tickets on every run to find similar ones; a real deployment would store those vectors (see the `E5VectorStore` sample). ML.NET cursors take no cancellation token, and neither transform can be saved with the model; see the `Starlings.Models.E5.MLNet` package README. e5 truncates text beyond 512 tokens; GLiNER2 rejects it. Scores can vary slightly across hardware.

Invalid arguments return exit code 2. A missing model cache, a missing `triage-model.zip` and inference errors return 1. This sample has been run on Linux x64 CPU only; platform support per model profile is in the [compatibility matrix](../../../docs/catalog/compatibility-matrix.md).
