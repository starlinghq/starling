# Derived-artifact cache

Status: **experimental**. Writer: `model-pipeline/src/model_pipeline/interventions/cache.py` (Python, tests in `model-pipeline/tests/test_intervention_cache.py`). Reader: `InterventionArtifactCache`, `InterventionArtifact` and `InterventionCacheKey` in `ai-dotnet/src/Starlings.Interventions` (tests in `InterventionArtifactTests.cs`). Design: [architecture.md](architecture.md); file formats: [profile-format.md](profile-format.md); compiler: [compiler.md](compiler.md).

A derived graph is compiled once per combination of base graph, profile and compiler, then reused. Requests never compile: they only change the strength input.

## Layout

```
<cache>/<model>/interventions/
    <key>/                          one entry; never modified after it is published
        model.onnx                  the derived graph
        model.onnx_data             hard link to the base graph's external data (one per external location)
        profile.json                the compiled profile, byte for byte
        directions/...              the profile's directions files, at their profile-relative paths
        compiler-manifest.json
    <key>.lock                      advisory lock file of every compile of <key>
    .tmp-<key>-<pid>-<random>/      an entry being built
```

- `<model>` is a cache model name, `[a-z0-9][a-z0-9._-]{0,63}` (for example `lfm3`).
- `<key>` is the cache key below: 64 lower-case hex characters. The acquisition store under the same cache root treats only `<owner>/<name>/<40-hex>` directories as snapshots, so its maintenance never lists or prunes an entry.
- External locations, `profile.json`, its directions files, `model.onnx` and `compiler-manifest.json` must not collide (equal paths, or one a directory prefix of another); the writer rejects a profile whose directions path would, before compiling.
- Lock files stay behind; they are empty and harmless.

## Cache key

The key is the SHA-256 of this UTF-8 text: one field per line, fields separated by one tab (U+0009), every line ending in LF (U+000A), nothing else.

```
starling-intervention-cache-key 1
compiler_version	<compiler_version>
site_mapping_version	<site_mapping_version>
opset	<opset, decimal>
precision	<precision>
profile_sha256	<profile_sha256>
base	<role>	<path>	<sha256>	<size, decimal>          one line per base artifact, sorted by path
input	<name>	<role>	<dtype>	[<d0>,<d1>,...]            one line per declared input, sorted by name
```

- Every value is the manifest field of the same name; `role` and `dtype` use their JSON spellings, and a shape is its dimensions in decimal, comma-separated, in brackets (`[1]`, or `[]` for a scalar).
- Sorting is ordinal over the UTF-8 bytes of the path (or name), so it does not depend on document order or on a language's string comparison.
- The contract forbids control characters in every component, so neither separator can occur inside a value.

Every component is in the compiler manifest, so `InterventionCacheKey.Compute(manifest)` (C#) and `manifest_cache_key(manifest)` (Python) recompute it from the manifest alone, and a reader checks it against the directory name. The writer computes the same key *before* compiling, from the base graph bytes, the external data identities, the profile hash, the site mapping, the compiler version and the strength input, so a repeat call finds the entry without compiling.

Example (the value both test suites check):

```
starling-intervention-cache-key 1
compiler_version	0.1.0
site_mapping_version	lfm3/fp32/1
opset	17
precision	fp32
profile_sha256	2222222222222222222222222222222222222222222222222222222222222222
base	graph	onnx/model.onnx	0000000000000000000000000000000000000000000000000000000000000000	1234
base	external_data	onnx/model.onnx_data	1111111111111111111111111111111111111111111111111111111111111111	4096
input	intervention_strength	strength	float32	[1]
```

has key `560773f98c03601aa3bd35112beebdbd3b91198e8dbac03f94eb1134499e6579`.

## Writing an entry (Python)

```python
from model_pipeline.interventions import get_or_compile

entry = get_or_compile(
    cache_root="~/.cache/starlings",       # same file system as the bundle (hard links)
    model="lfm3",
    base_dir=snapshot_dir,                 # the base model directory; base_artifacts paths are relative to it
    graph="onnx/model.onnx",
    profile="profiles/synthetic/profile.json",
    external_sha256={"onnx/model.onnx_data": lock_sha},   # optional: identities the caller already trusts
)
entry.graph_path, entry.key, entry.created
```

1. Hash the base graph bytes; the profile's `base_model_sha256` must equal it, or nothing is written. Find the site mapping for the hash (an unknown hash is unsupported).
2. Compute the key. External data identities come from `external_sha256` where given (for example the asset lock's SHA-256), otherwise each file is hashed.
3. If `<key>/` exists, open and verify it (below) and return it. A failure raises `InterventionCacheError` and leaves the entry in place: a corrupt entry is never silently replaced. `evict(cache_root, model, key)` removes it (renamed away under the lock first, then deleted; removing a hard link never touches the original). If the entry disappeared while it was being read (a concurrent `evict`), the call compiles it again under the lock instead of reporting a verification failure.
4. Otherwise take an exclusive `flock` on `<key>.lock` (default timeout 600 s, then `TimeoutError`), and check again: another process may have published it while this one waited.
5. Remove any `.tmp-<key>-*` directory: every compile of the key holds the lock, so one found now is from a crashed process. Temporary directories of other keys are left alone.
6. Compile into a new `.tmp-<key>-<pid>-<random>/` beside the final directory (same parent, so the rename is atomic): `compile_intervention` writes `model.onnx`, hard-links the external data and writes the manifest, recording each link's device, inode and modification time (`derived_files[].identity`) and how onnx.checker judged the graph (`validation`). The key recomputed from the new manifest must equal the key from step 2; a wrong `external_sha256` or a base file that changed fails here. Copy `profile.json` verbatim and its directions files, fsync, verify the temporary entry, and rename it to `<key>/`.
7. Any failure removes the temporary directory and publishes nothing.

The source bundle and the profile directory are only read. Weights are never copied.

## External data and file systems

The derived graph reaches the original weights through hard links next to `model.onnx`, under the base graph's own location names. That is the only route ONNX Runtime 1.30 accepted in the spike (C# and Python): parent-relative and absolute locations, and file or directory symlinks, are all rejected because ORT requires the resolved path to stay inside the model's directory. So:

- **The cache must be on the bundle's file system.** Across file systems a hard link is impossible, and the writer raises `ExternalDataLinkError` (`errno` `EXDEV`). The message names both mount points and the remedy: put `cache_root` on the bundle's file system (for example under the bundle's mount point) or acquire the bundle onto the cache's file system. The compiler checks the device numbers (`st_dev`) of the external files against the output directory *before* it writes anything; a link that still fails with `EXDEV` (two bind mounts of one file system share a device number) raises the same error, and the links made so far are removed.
- **Decision: fail, no copy fallback.** A copy would be a second full set of weights per entry (7 GB for LFM3 fp32) made silently on a configuration mistake, and an entry that is a copy would need its own verification rules (it is not the inode of the base file). Failing early with the remedy keeps one invariant: the weights exist once. The other measured route, the session option `session.model_external_initializers_file_folder_path`, needs session-level configuration and is not implemented.
- **A link shares the inode with the original.** Replacing the base file by rename leaves the link on the old bytes; rewriting it in place changes what the derived graph reads. The linked files are listed in the manifest's `derived_files` with their SHA-256, size and identity (device, inode, modification time), so Fast verification catches a replaced link and a rewrite that changes the modification time, and Full verification catches any byte change.

## Verification

Both sides offer two levels. **Fast** is for every load; **Full** is for when the bytes must be proven, for example after a crash, a file-system repair, or a change to the bundle.

| Check | Fast | Full |
| --- | --- | --- |
| Manifest parses (strict, schema version 1) | yes | yes |
| Key recomputed from the manifest equals the directory name (and the expected key) | yes | yes |
| `profile.json` SHA-256 equals `profile_sha256`; id, operator, base graph and sites match; directions files verify (size, SHA-256, finite) | yes | yes |
| `model.onnx` size and SHA-256 | yes | yes |
| Each other derived file is the link of a base `external_data` artifact (path relative to the base graph's directory) with that artifact's SHA-256 and size, and every base external data file has its link | yes | yes |
| Each other derived file (the linked weights): exists with the recorded size | yes | yes |
| Each other derived file has the recorded `identity`: device, inode and modification time (ns); a manifest without it fails | yes | — |
| With the base directory: each linked file is still the same inode as its base file | yes | — |
| SHA-256 of each other derived file (reads the whole weights file) | no | yes |
| With the base directory: size and SHA-256 of every base artifact | no | yes |

- A weights file rewritten in place changes its modification time and fails Fast. Only a rewrite that keeps the size and restores the modification time (for example `touch -d`) passes Fast; Full catches it.
- The identity is machine-local. A copied, restored or re-acquired entry or bundle (new inodes, identical bytes) fails Fast and passes Full; evict and recompile it to make Fast pass again. Full does not rewrite the manifest. Device numbers are not stable everywhere: on btrfs subvolumes, overlayfs, NFS or a device-mapper volume they can change across a remount or reboot, and every Fast open of the entry then fails the same way (this repository's development cache is on btrfs; not observed so far).
- Python reads the identity with `os.stat`. C# reads it on Linux with `statx(2)` (device as glibc encodes `st_dev`, the same numbers Python reports; checked across the two languages on one entry). On other platforms, or a Linux without `statx`, the C# Fast level compares the modification time only, at the 100 ns precision .NET reports, and skips the base-directory inode check. Windows and macOS were not tested.
- An entry written before identities were recorded fails Fast with "records no identity"; Full still opens it. No released writer produced such entries (the cache is unreleased), so the compiler version did not change: the derived graph bytes are the same.

The cache key covers the compile inputs (`base_artifacts`, profile, versions, inputs), not `derived_graph` or `derived_files`; the rule that each derived file carries its base artifact's identity is what ties the bytes ONNX Runtime reads to the key. `compiler-manifest.json` itself is the local root of trust: it is not signed, so these checks detect corruption, partial writes and changed bundles, not a deliberate rewrite of a whole entry by someone who can write to the cache directory.

```csharp
InterventionArtifact artifact = InterventionArtifactCache.Open(cacheRoot, "lfm3", key);       // Fast
artifact.Manifest.VerifyBaseModel(expectedBaseGraphSha256);  // before any session is created
// artifact.GraphPath loads with OnnxModelSession.Create(path, options); inputs: base inputs + artifact.Manifest.Inputs
```

`InterventionArtifact.Open(directory)` opens any directory with the cache-entry layout (manifest, `model.onnx`, linked data, `profile.json` and its directions files) without the name check; plain `compile_intervention` output has no `profile.json`, so it does not open; `InterventionArtifactCache.ListEntries(cacheRoot, model)` lists published entries (key-named directories only) without opening them.

## Not covered

- Windows: the writer's lock uses POSIX `flock`; NTFS hard links and the C# reader on Windows were not tested.
- Cross-file-system caches (fail before anything is written, with the remedy in the message; no copy fallback, see above).
- Eviction policy, size limits and garbage collection of unused entries: entries stay until `evict` removes them.
- The concurrency tests use threads in one process (separate `flock` descriptions); a multi-process race was not run.
- Real models: no site mapping ships yet, so every real graph is unsupported.
