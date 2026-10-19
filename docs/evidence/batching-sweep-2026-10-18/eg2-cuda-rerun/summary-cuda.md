# Batching sweep summary

2 run(s), 56 cell(s).
Skipped by the in-flight guard (not measured): embeddinggemma-2/cuda/long/b32/c16, embeddinggemma-2/cuda/long/b64/c16, embeddinggemma-2/cuda/mixed/b32/c16, embeddinggemma-2/cuda/mixed/b64/c16, embeddinggemma-2/cuda/short/b32/c16, embeddinggemma-2/cuda/short/b64/c16.

Items/s = items completed / cell wall time (all callers). Latency is per call. Padding efficiency = real tokens / padded cells (rows x longest row per ORT run, as the library chunks). Quality: every output vs the same item run alone, worst difference against the profile's padded-batch tolerance. Micro-batched configurations (micro-wW-rR: MaxWait W ms, MaxRows R) issue every item as its own call through the library's batcher; their padding is the dispatched batches', and 'micro rows / queue ms' is the mean rows per dispatched batch and the mean time an item waited in the queue.

## embeddinggemma-2 · cuda · baseline · intra-op 4 · max_batch_size call (64) · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 49.5 | 67.3 | 69.5 |
| 4 | 59.8 | 72.2 | 70.9 |
| 8 | 61.9 | 75.0 | 73.7 |
| 16 | 64.5 | 71.2 | 65.2 |
| 32 | 64.5 | 71.0 |  |
| 64 | 64.2 | 71.1 |  |

Best: batch 8 at concurrency 4, 75.0 items/s; best single-item callers: concurrency 16, 69.5 items/s (ratio 1.08).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 99 | 49.5 | 20.19 | 21.65 | 1.000 | tokenize 0.73, encode 19.44 | 1274 | 3582 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 137 | 67.3 | 59.26 | 62.42 | 1.000 | tokenize 0.92, encode 58.39 | 1281 | 3598 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 148 | 69.5 | 221.95 | 284.81 | 1.000 | tokenize 1.20, encode 219.80 | 1273 | 4718 |  | 0.00e+00 (5e-06) | 0 | pass |
| 4 | 1 | 1 | 30 | 59.8 | 67.50 | 68.99 | 0.921 | tokenize 2.59, encode 64.90 | 1277 | 4710 |  | 2.01e-07 (5e-06) | 0 | pass |
| 4 | 4 | 1 | 38 | 72.2 | 220.80 | 234.68 | 0.921 | tokenize 2.91, encode 217.04 | 1280 | 4622 |  | 2.01e-07 (5e-06) | 0 | pass |
| 4 | 16 | 1 | 47 | 70.9 | 846.72 | 1035.00 | 0.920 | tokenize 8.07, encode 839.26 | 1482 | 8814 |  | 2.01e-07 (5e-06) | 0 | pass |
| 8 | 1 | 1 | 16 | 61.9 | 129.85 | n/a | 0.907 | tokenize 5.92, encode 123.63 | 1485 | 8806 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 21 | 75.0 | 428.78 | 490.39 | 0.906 | tokenize 7.21, encode 419.25 | 1465 | 8718 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 30 | 73.7 | 1677.98 | 2639.43 | 0.908 | tokenize 17.74, encode 1652.61 | 1774 | 13934 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 64.5 | 248.66 | n/a | 0.900 | tokenize 9.95, encode 238.51 | 1798 | 13926 |  | 2.60e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 71.2 | 888.80 | n/a | 0.901 | tokenize 12.60, encode 875.60 | 1893 | 13838 |  | 2.60e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 16 | 65.2 | 3261.70 | n/a | 0.900 | tokenize 48.07, encode 3215.74 | 2240 | 13934 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 64.5 | 496.40 | n/a | 0.900 | tokenize 18.99, encode 477.19 | 2240 | 13822 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 71.0 | 1793.44 | n/a | 0.900 | tokenize 30.09, encode 1763.10 | 2241 | 13838 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 64.2 | 996.43 | n/a | 0.900 | tokenize 38.08, encode 957.32 | 2236 | 13822 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 71.1 | 3615.54 | n/a | 0.900 | tokenize 58.07, encode 3563.13 | 2156 | 13838 |  | 2.60e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 63.3 | 129.6 | 143.6 |
| 4 | 79.3 | 94.7 | 95.4 |
| 8 | 72.1 | 83.9 | 86.2 |
| 16 | 69.4 | 76.2 | 70.4 |
| 32 | 68.4 | 74.9 |  |
| 64 | 69.3 | 75.8 |  |

Best: batch 1 at concurrency 16, 143.6 items/s; best single-item callers: concurrency 16, 143.6 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 127 | 63.3 | 14.63 | 21.48 | 1.000 | tokenize 0.28, encode 14.16 | 1859 | 13814 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 264 | 129.6 | 29.60 | 47.31 | 1.000 | tokenize 0.17, encode 29.19 | 1812 | 13838 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 301 | 143.6 | 110.59 | 139.49 | 1.000 | tokenize 0.25, encode 110.42 | 1864 | 13934 |  | 0.00e+00 (5e-06) | 0 | pass |
| 4 | 1 | 1 | 40 | 79.3 | 56.24 | 71.84 | 0.482 | tokenize 0.95, encode 54.87 | 1874 | 13830 |  | 2.58e-07 (5e-06) | 0 | pass |
| 4 | 4 | 1 | 48 | 94.7 | 172.90 | 192.64 | 0.473 | tokenize 1.12, encode 169.42 | 1780 | 13838 |  | 2.58e-07 (5e-06) | 0 | pass |
| 4 | 16 | 1 | 57 | 95.4 | 609.89 | 1171.07 | 0.473 | tokenize 1.34, encode 609.30 | 1851 | 13934 |  | 2.58e-07 (5e-06) | 0 | pass |
| 8 | 1 | 1 | 19 | 72.1 | 108.46 | n/a | 0.416 | tokenize 2.03, encode 106.46 | 1851 | 13926 |  | 3.57e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 23 | 83.9 | 353.97 | 458.72 | 0.401 | tokenize 2.13, encode 351.84 | 1781 | 13838 |  | 3.57e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 30 | 86.2 | 1438.82 | 2760.29 | 0.408 | tokenize 11.28, encode 1434.67 | 1817 | 13934 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 69.4 | 226.15 | n/a | 0.359 | tokenize 4.39, encode 218.26 | 1818 | 13926 |  | 2.60e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 11 | 76.2 | 767.19 | n/a | 0.359 | tokenize 5.45, encode 760.41 | 1923 | 13838 |  | 2.60e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 16 | 70.4 | 2522.37 | n/a | 0.367 | tokenize 14.50, encode 2507.30 | 2283 | 13934 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 68.4 | 441.97 | n/a | 0.375 | tokenize 8.06, encode 431.97 | 2285 | 13814 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 74.9 | 1468.22 | n/a | 0.375 | tokenize 9.27, encode 1436.21 | 2393 | 13838 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 69.3 | 991.16 | n/a | 0.367 | tokenize 15.28, encode 908.34 | 2326 | 13814 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 75.8 | 3020.99 | n/a | 0.367 | tokenize 17.54, encode 2969.21 | 2371 | 13838 |  | 2.60e-07 (5e-06) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 75.8 | 183.0 | 220.5 |
| 4 | 307.9 | 733.5 | 782.0 |
| 8 | 591.4 | 1020.8 | 1084.7 |
| 16 | 880.4 | 1177.3 | 1208.9 |
| 32 | 1022.4 | 1249.2 |  |
| 64 | 1058.7 | 1273.7 |  |

Best: batch 64 at concurrency 4, 1273.7 items/s; best single-item callers: concurrency 16, 220.5 items/s (ratio 5.78).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 152 | 75.8 | 12.98 | 14.98 | 1.000 | tokenize 0.12, encode 12.83 | 1134 | 2338 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 370 | 183.0 | 22.66 | 28.70 | 1.000 | tokenize 0.06, encode 22.59 | 1141 | 2412 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 452 | 220.5 | 68.83 | 97.85 | 1.000 | tokenize 0.06, encode 68.74 | 1156 | 2670 |  | 0.00e+00 (5e-06) | 0 | pass |
| 4 | 1 | 1 | 154 | 307.9 | 12.93 | 13.63 | 0.813 | tokenize 0.10, encode 12.82 | 1156 | 2550 |  | 1.74e-07 (5e-06) | 0 | pass |
| 4 | 4 | 1 | 370 | 733.5 | 21.19 | 24.59 | 0.815 | tokenize 0.12, encode 21.03 | 1156 | 2574 |  | 1.74e-07 (5e-06) | 0 | pass |
| 4 | 16 | 1 | 402 | 782.0 | 74.70 | 104.94 | 0.815 | tokenize 0.19, encode 74.51 | 1158 | 2670 |  | 1.74e-07 (5e-06) | 0 | pass |
| 8 | 1 | 1 | 148 | 591.4 | 13.55 | 14.04 | 0.757 | tokenize 0.20, encode 13.33 | 1158 | 2550 |  | 1.13e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 258 | 1020.8 | 31.42 | 32.50 | 0.757 | tokenize 0.22, encode 31.18 | 1162 | 2574 |  | 1.13e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 284 | 1084.7 | 117.01 | 150.39 | 0.756 | tokenize 0.41, encode 116.63 | 1194 | 3694 |  | 1.13e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 111 | 880.4 | 18.18 | 19.17 | 0.700 | tokenize 0.37, encode 17.83 | 1189 | 3574 |  | 2.15e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 151 | 1177.3 | 53.97 | 56.84 | 0.700 | tokenize 0.40, encode 53.57 | 1189 | 3598 |  | 2.15e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 160 | 1208.9 | 203.47 | 274.32 | 0.700 | tokenize 0.83, encode 202.25 | 1204 | 3694 |  | 2.15e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 64 | 1022.4 | 30.66 | 33.30 | 0.654 | tokenize 0.77, encode 29.88 | 1209 | 3590 |  | 1.64e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 80 | 1249.2 | 101.56 | 106.04 | 0.654 | tokenize 0.92, encode 100.61 | 1229 | 3598 |  | 1.64e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 34 | 1058.7 | 62.05 | 63.06 | 0.644 | tokenize 1.88, encode 59.04 | 1242 | 3590 |  | 1.86e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 41 | 1273.7 | 200.24 | 215.40 | 0.644 | tokenize 1.81, encode 197.67 | 1274 | 3598 |  | 1.86e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · baseline · intra-op 4 · max_batch_size default · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 61.3 |
| 16 | 63.2 |
| 32 | 63.9 |
| 64 | 64.3 |

Best: batch 64 at concurrency 1, 64.3 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 16 | 61.3 | 129.58 | n/a | 0.907 | tokenize 5.79, encode 123.60 | 1157 | 3362 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 63.2 | 251.64 | n/a | 0.900 | tokenize 12.16, encode 238.74 | 1157 | 3362 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 63.9 | 500.63 | n/a | 0.900 | tokenize 20.78, encode 477.74 | 1157 | 3362 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 64.3 | 997.31 | n/a | 0.900 | tokenize 37.73, encode 955.23 | 1157 | 3362 |  | 2.60e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 73.0 |
| 16 | 70.5 |
| 32 | 68.3 |
| 64 | 68.6 |

Best: batch 8 at concurrency 1, 73.0 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 19 | 73.0 | 106.90 | n/a | 0.416 | tokenize 1.87, encode 104.54 | 1153 | 3362 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 70.5 | 221.50 | n/a | 0.359 | tokenize 3.94, encode 217.71 | 1157 | 3362 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 68.3 | 439.01 | n/a | 0.370 | tokenize 8.31, encode 431.30 | 1156 | 3362 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 68.6 | 989.81 | n/a | 0.364 | tokenize 17.40, encode 916.44 | 1157 | 3362 |  | 2.60e-07 (5e-06) | 0 | pass |


## Batching configurations (items/s)

Best valid cell of each configuration, and single-item callers (batch 1) at each concurrency. A micro-batched configuration is adopted only if its batch-1 rows beat the baseline's batch-1 rows at concurrency 4 and 16.

### embeddinggemma-2 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · long

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 75.0 | (8, 4) | 0.906 | 428.78 | 49.5 | 67.3 | 69.5 |
| baseline (max_batch_size default) | 64.3 | (64, 1) | 0.900 | 997.31 |  |  |  |

### embeddinggemma-2 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · mixed

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 143.6 | (1, 16) | 1.000 | 110.59 | 63.3 | 129.6 | 143.6 |
| baseline (max_batch_size default) | 73.0 | (8, 1) | 0.416 | 106.90 |  |  |  |
