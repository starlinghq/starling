| Row | Tokens | Expect | C# outcome | Ids equal | Embedding max abs | Cosine | Seconds |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `txt-alice-2600-none` | 2600 | embed | embedded | yes | 6.00e-08 | 1.000000000000 | 9.42 |
| `txt-alice-4096-query` | 4096 | embed | embedded | yes | 6.46e-08 | 1.000000000000 | 12.12 |
| `txt-alice-6144-document` | 6144 | embed | embedded | yes | 7.50e-08 | 1.000000000000 | 25.86 |
| `txt-alice-8191-document` | 8191 | embed | embedded | yes | 6.00e-08 | 1.000000000000 | 47.74 |
| `txt-alice-8192-none` | 8192 | embed | embedded | yes | 7.00e-08 | 1.000000000000 | 43.52 |
| `txt-alice-8192-query` | 8192 | embed | embedded | yes | 4.70e-08 | 1.000000000000 | 43.61 |
| `txt-constitution-document` | 5950 | embed | embedded | yes | 8.30e-08 | 1.000000000000 | 23.94 |
| `txt-corpus-cycle-8192-none` | 8192 | embed | embedded | yes | 1.00e-07 | 1.000000000000 | 43.8 |
| `txt-alice-8193-none` | 8193 | reject | rejected (EmbeddingGemma2InputTooLongException, 8193 tokens) | n/a (rejected) |  |  | 0.01 |
| `txt-alice-8192-text-document-prompt` | 8198 | reject | rejected (EmbeddingGemma2InputTooLongException, 8198 tokens) | n/a (rejected) |  |  | 0.01 |
| `mm-alice-6-images-6000-none` | 6000 | embed | embedded | yes | 2.82e-07 | 0.999999999997 | 49.61 |
| `mm-alice-8-images-8192-document` | 8192 | embed | embedded | yes | 2.66e-07 | 0.999999999998 | 80.19 |
| `mm-alice-6-images-8193-none` | 8193 | reject | rejected (EmbeddingGemma2InputTooLongException, 8193 tokens) | n/a (rejected) |  |  | 0.01 |
| `mm-alice-video32-7000-none` | 7000 | embed | embedded | yes | 1.50e-07 | 0.999999999999 | 91.46 |
| `mm-alice-video32-8192-query` | 8192 | embed | embedded | yes | 6.78e-07 | 0.999999999987 | 104.09 |
| `mm-alice-two-videos-image-8192-document` | 8192 | embed | embedded | yes | 2.86e-07 | 0.999999999997 | 129.6 |
| `mm-alice-video32-8193-none` | 8193 | reject | rejected (EmbeddingGemma2InputTooLongException, 8193 tokens) | n/a (rejected) |  |  | 0.01 |

- Worst (text rows, 8 embedded): embedding max abs 1.00e-07 (`txt-corpus-cycle-8192-none`), min cosine 1.000000000000
- Worst (media rows, 5 embedded): embedding max abs 6.78e-07 (`mm-alice-video32-8192-query`), min cosine 0.999999999987
- Worst (all rows, 13 embedded): embedding max abs 6.78e-07 (`mm-alice-video32-8192-query`), min cosine 0.999999999987
- Padded batch `text-2600-and-8192` (txt-alice-2600-none, txt-alice-8192-none) vs C# single rows: max abs 0.00e+00
- Padded batch `six-images-and-32-frame-video` (mm-alice-6-images-6000-none, mm-alice-video32-7000-none) vs C# single rows: max abs 0.00e+00
