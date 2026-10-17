# Decision development parity: kev-0.8b/onnx-fp32-cpu

Status: **passed**; 30 records, 58 questions, 16 padded batches. Policy status: **proposed**.

| Check | Bound | Max abs | Mean abs | p99 abs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| python-onnx-cached/vs_reference | 2.0e-04 | 2.852e-05 | 6.469e-06 | 2.568e-05 | passed |
| python-onnx-uncached/vs_reference | 2.0e-04 | 2.852e-05 | 6.453e-06 | 2.548e-05 | passed |
| dotnet-ids-cached/vs_reference | 2.0e-04 | 2.852e-05 | 6.481e-06 | 2.575e-05 | passed |
| dotnet-ids-uncached/vs_reference | 2.0e-04 | 2.852e-05 | 6.455e-06 | 2.548e-05 | passed |
| dotnet-text/vs_reference | 2.0e-04 | 2.852e-05 | 6.481e-06 | 2.575e-05 | passed |
| python-onnx/cached_vs_uncached | 1.0e-05 | 1.848e-06 | 3.554e-07 | 1.338e-06 | passed |
| dotnet/cached_vs_uncached | 1.0e-05 | 1.940e-06 | 3.453e-07 | 1.404e-06 | passed |
| python-onnx/padded_batch_vs_single | 1.0e-05 | 2.146e-06 | 4.555e-07 | 2.095e-06 | passed |
| dotnet-ids-cached/vs_python-onnx-cached | 1.0e-06 | 1.563e-07 | 4.966e-08 | 1.447e-07 | passed |
| dotnet-ids-uncached/vs_python-onnx-uncached | 1.0e-06 | 1.361e-07 | 5.269e-08 | 1.355e-07 | passed |
| dotnet-text/vs_python-onnx-cached | 1.0e-06 | 1.563e-07 | 4.966e-08 | 1.447e-07 | passed |

Argmax flips against the reference (outside / listed near ties):

- python-onnx-cached: 0 / 0
- python-onnx-uncached: 0 / 0
- dotnet-ids-cached: 0 / 0
- dotnet-ids-uncached: 0 / 0
- dotnet-text: 0 / 0
- python-onnx-padded-batch: 0 / 0

Accuracy on 44 labelled questions: identical.
Token ids (.NET against reference): identical.
Non-finite outputs: 0.

Mean and p99 are report-only, over per-question maximum absolute differences. Protected G1/G2 release qualification is **unapproved**; no protected fixtures were adopted.
