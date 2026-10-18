# E5VectorStore sample

Local multilingual search using the published Starling packages and `Microsoft.Extensions.VectorData`. Requires the .NET 10 SDK. The in-memory connector is `CommunityToolkit.VectorData.InMemory`; it belongs only to this sample, not the Starling libraries.

From this directory:

```sh
dotnet build -c Release
dotnet run -c Release --no-build -- download
dotnet run -c Release --no-build -- search "Which river flows through Cologne?"
```

`download` explicitly acquires and verifies the pinned model (about 487 MB). `search` verifies the local cache and runs offline. Both accept `--cache <dir>`; otherwise they use `$STARLING_MODELS` or the user cache (`~/.cache/starlings/models` on Linux). A missing or corrupt cache fails rather than downloading during search. `StarlingsVersion` pins the published package version; it is `0.1.0-preview.6`, the version this sample was written and run with, not the current `0.2.0`. The sample uses `PackageReference`s only, no project references, and needs `Microsoft.ML.OnnxRuntime` 1.30.0 (CPU). `packages.lock.json` records the dependency graph. To verify it, use `dotnet restore --locked-mode`.

`E5EmbeddingGenerator` is the collection's embedding generator. The record's string-valued `[VectorStoreVector(384)]` property provides `passage: ` text; upsert automatically embeds it. `SearchAsync` receives a `query: ` prefixed question and automatically embeds that too. Neither step computes vectors manually. The generator's default input type leaves these explicit prefixes alone. Cosine similarity ranks the results, with larger scores first.

See Microsoft's [embedding-generation documentation](https://learn.microsoft.com/semantic-kernel/concepts/vector-store-connectors/embedding-generation) and [in-memory connector documentation](https://learn.microsoft.com/semantic-kernel/concepts/vector-store-connectors/out-of-the-box-connectors/inmemory-connector) for the collection contract. The collection is recreated each run and is not persistent storage.

Actual stdout from Linux x64 CPU, .NET SDK 10.0.100, runtime 10.0.12, Starlings 0.1.0-preview.6 and connector 1.0.1, on 2026-10-17 (the command above, using an already populated default cache):

```text
Indexed 4 multilingual passages in memory.
Query: Which river flows through Cologne?
0.8186  rhine: Der Rhein fließt durch Köln.
0.7177  everest: Mount Everest is the highest mountain above sea level.
0.6889  tokyo: 東京は日本の首都です。
```

Small floating-point differences across hardware are possible. This sample has been run on Linux x64 CPU only; see the repository's [compatibility matrix](../../../docs/catalog/compatibility-matrix.md) for model profiles. An invalid command returns exit code 2; an acquisition or inference error returns 1.
