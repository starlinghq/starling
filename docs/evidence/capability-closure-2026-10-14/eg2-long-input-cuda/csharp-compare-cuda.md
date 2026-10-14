| Row | Tokens | Expect | C# outcome | Ids equal | Embedding max abs | Cosine | Seconds |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `txt-alice-2600-none` | 2600 | embed | embedded | yes | 9.70e-08 | 1.000000000000 | 5.38 |
| `txt-alice-4096-query` | 4096 | embed | embedded | yes | 8.50e-08 | 1.000000000000 | 0.3 |
| `txt-alice-6144-document` | 6144 | embed | embedded | yes | 8.90e-08 | 1.000000000000 | 0.54 |
| `txt-alice-8191-document` | 8191 | embed | embedded | yes | 9.50e-08 | 1.000000000000 | 0.8 |
| `txt-alice-8192-none` | 8192 | embed | embedded | yes | 7.00e-08 | 1.000000000000 | 0.81 |
| `txt-alice-8192-query` | 8192 | embed | embedded | yes | 8.46e-08 | 1.000000000000 | 0.81 |
| `txt-constitution-document` | 5950 | embed | embedded | yes | 1.29e-07 | 1.000000000000 | 0.46 |
| `txt-corpus-cycle-8192-none` | 8192 | embed | embedded | yes | 1.98e-07 | 0.999999999999 | 0.81 |
| `txt-alice-8193-none` | 8193 | reject | rejected (EmbeddingGemma2InputTooLongException, 8193 tokens) | n/a (rejected) |  |  | 0.02 |
| `txt-alice-8192-text-document-prompt` | 8198 | reject | rejected (EmbeddingGemma2InputTooLongException, 8198 tokens) | n/a (rejected) |  |  | 0.02 |
| `mm-alice-6-images-6000-none` | 6000 | embed | embedded | yes | 6.51e-07 | 0.999999999990 | 1.39 |
| `mm-alice-8-images-8192-document` | 8192 | embed | embedded | yes | 2.36e-07 | 0.999999999998 | 2.03 |
| `mm-alice-6-images-8193-none` | 8193 | reject | rejected (EmbeddingGemma2InputTooLongException, 8193 tokens) | n/a (rejected) |  |  | 0.02 |
| `mm-alice-video32-7000-none` | 7000 | embed | embedded | yes | 2.27e-07 | 0.999999999999 | 2.33 |
| `mm-alice-video32-8192-query` | 8192 | embed | embedded | yes | 3.54e-07 | 0.999999999997 | 2.5 |
| `mm-alice-two-videos-image-8192-document` | 8192 | embed | embedded | yes | 3.95e-07 | 0.999999999996 | 3.17 |
| `mm-alice-video32-8193-none` | 8193 | reject | rejected (EmbeddingGemma2InputTooLongException, 8193 tokens) | n/a (rejected) |  |  | 0.02 |

- Worst (text rows, 8 embedded): embedding max abs 1.98e-07 (`txt-corpus-cycle-8192-none`), min cosine 0.999999999999
- Worst (media rows, 5 embedded): embedding max abs 6.51e-07 (`mm-alice-6-images-6000-none`), min cosine 0.999999999990
- Worst (all rows, 13 embedded): embedding max abs 6.51e-07 (`mm-alice-6-images-6000-none`), min cosine 0.999999999990
- Padded batch `text-2600-and-8192` (txt-alice-2600-none, txt-alice-8192-none): not run ([ErrorCode:Fail] Non-zero status code returned while running MatMul node. Name:'node_MatMul_203' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:360 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Failed to allocate memory for requested buffer of size 2147483648)
- Padded batch `six-images-and-32-frame-video` (mm-alice-6-images-6000-none, mm-alice-video32-7000-none): not run ([ErrorCode:Fail] Non-zero status code returned while running FusedMatMul node. Name:'mm_ple_linear/MatMulScaleFusion/' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:360 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Failed to allocate memory for requested buffer of size 688128000)
