# Decision development parity: kev-4b/onnx-fp32-cpu

Status: **passed**; 30 records, 58 questions, 16 padded batches. Policy status: **proposed**.

| Check | Bound | Max abs | Mean abs | p99 abs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| python-onnx-cached/vs_reference | 2.0e-04 | 6.366e-05 | 7.576e-06 | 4.669e-05 | passed |
| python-onnx-uncached/vs_reference | 2.0e-04 | 6.557e-05 | 7.626e-06 | 4.761e-05 | passed |
| dotnet-ids-cached/vs_reference | 2.0e-04 | 6.366e-05 | 7.572e-06 | 4.669e-05 | passed |
| dotnet-ids-uncached/vs_reference | 2.0e-04 | 6.568e-05 | 7.622e-06 | 4.762e-05 | passed |
| dotnet-text/vs_reference | 2.0e-04 | 6.366e-05 | 7.572e-06 | 4.669e-05 | passed |
| python-onnx/cached_vs_uncached | 1.0e-05 | 1.937e-06 | 2.874e-07 | 1.343e-06 | passed |
| dotnet/cached_vs_uncached | 1.0e-05 | 2.030e-06 | 2.771e-07 | 1.403e-06 | passed |
| python-onnx/padded_batch_vs_single | 1.0e-05 | 2.503e-06 | 3.550e-07 | 2.196e-06 | passed |
| dotnet-ids-cached/vs_python-onnx-cached | 1.0e-06 | 1.452e-07 | 4.590e-08 | 1.450e-07 | passed |
| dotnet-ids-uncached/vs_python-onnx-uncached | 1.0e-06 | 1.485e-07 | 5.598e-08 | 1.478e-07 | passed |
| dotnet-text/vs_python-onnx-cached | 1.0e-06 | 1.452e-07 | 4.590e-08 | 1.450e-07 | passed |

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
