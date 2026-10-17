# Kev soaks and a RunPod CUDA check, 2026-10-17

Development evidence for the four Kev profiles: one-hour `Starlings.Soak.Tool` runs (plan 10.2, the G4 procedure) on the
workstation, and the Kev CUDA tests plus 10-minute soak smokes on a RunPod RTX 3090 from the public repository and the
public Hugging Face bundles. Not gate results: Kev's references are proposed and no gate is recorded in the manifests.

Each soak cycle loads the model, serves concurrent requests (two callers; every 10th request is a long request with a
5 to 150 ms deadline), checks every output against the first cycle's, disposes the model and samples memory. The
growth rule is the tool's own, unchanged.

## One-hour soaks (workstation: Intel i7-5820K, RTX 3090 PCI device 0, Linux 7.2.5, .NET 10.0.12, ORT 1.30.0)

| Profile | Allocator setting | Cycles | Requests checked, wrong / errors | Request p50 | Verdict | Report |
| --- | --- | ---: | --- | ---: | --- | --- |
| `kev-0.8b/onnx-fp32-cuda` | default | 477 | 25,758, 0 / 0 | 175 ms | **FAIL**: loaded rss_anon +51 MB (64 MB/h), growing in both halves | [report](workstation/kev-0.8b-20261017T173435Z.md) |
| `kev-0.8b/onnx-fp32-cuda` | `MALLOC_MMAP_THRESHOLD_=65536` | 691 | 37,314, 0 / 0 | 134 ms | **PASS** | [report](workstation/kev-0.8b-20261017T193629Z.md) |
| `kev-4b/onnx-fp32-cuda` | default | 435 | 7,830, 0 / 0 | 363 ms | **PASS** (GPU memory flat at 17.7 GB) | [report](workstation/kev-4b-20261017T183518Z.md) |
| `kev-0.8b/onnx-fp32-cpu` | `MALLOC_MMAP_THRESHOLD_=65536` | 133 | 7,182, 0 / 0 | 900 ms | **PASS** | [report](workstation/kev-0.8b-20261017T203704Z.md) |
| `kev-4b/onnx-fp32-cpu` | `MALLOC_MMAP_THRESHOLD_=65536` | 56 | 1,008, 0 / 0 | 4,985 ms | **PASS** | [report](workstation/kev-4b-20261017T213724Z.md) |

The default-allocator 0.8B CUDA failure has the signature of `docs/evidence/gliner2-soak-growth/` (glibc arena
retention under concurrent native allocations; outputs correct throughout), and the documented Linux setting
`MALLOC_MMAP_THRESHOLD_=65536` removes it: the rerun plateaus over 691 cycles. That setting is the one these profiles
are soaked with, as for GLiNER2 and EmbeddingGemma 2 video. Caveats: builds and test suites ran on the same machine
during part of the first (failing) run, which affects its latencies, not its output checks; the 0.8B CPU run at the
default allocator was not completed (stopped when the run order changed), so the CPU profiles have no default-allocator
result.

## RunPod RTX 3090 (EU-CZ-1, secure cloud, `runpod/pytorch:1.4.0-cu1300-torch291-ubuntu2404`)

An agent pod (not the qualification workflow): NVIDIA GeForce RTX 3090, driver 580.159.03, AMD EPYC 7H12; .NET SDK
10.0.100; CUDA 13 runtime and cuDNN 9 from the pinned wheels of `model-pipeline/reference/embeddinggemma2/requirements-cuda.txt`
(the image's CUDA 13 runtime also present, [libraries](runpod/cuda-libs.txt)). Repository at `855ec53`, bundles
downloaded anonymously with `hf download` from `Starlinghq/kev-0.8b-onnx@8a9fe017` (14 s) and
`Starlinghq/kev-4b-onnx@8179da50` (41 s), every file equal to `SHA256SUMS` ([0.8B](runpod/sha-kev-0.8b.txt), [4B](runpod/sha-kev-4b.txt)).

| Check | Kev-0.8B | Kev-4B |
| --- | --- | --- |
| `Starlings.Models.Kev.Tests.Cuda` ([results](runpod/kev-cuda-tests.xml)) | 12 of 12 passing (both sizes) | |
| 9 request goldens vs PyTorch fp32 (2e-4) | 1.33e-5 | 4.14e-5 |
| Cached vs uncached (1e-5) | 8.6e-7 | 7.8e-7 |
| 4 callers x 2 rounds, serialized | 72 of 72 bit-identical | 72 of 72 |
| Cancelled calls leaving device work queued | 0 of 10 | 0 of 10 |
| One 8-question request | 89 ms | 267 ms |
| 10-minute soak smoke, `MALLOC_MMAP_THRESHOLD_=65536` | PASS, 122 cycles, 6,588 requests, 0 wrong ([report](runpod/kev-0.8b-20261017T215839Z.md)) | PASS, 76 cycles, 1,368 requests, 0 wrong ([report](runpod/kev-4b-20261017T220847Z.md)) |

Pod `h28juy4zjznm14`, about 28 minutes, $0.50/h; terminated and confirmed gone through the API.
