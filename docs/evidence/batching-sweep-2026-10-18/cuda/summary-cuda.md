# Batching sweep summary

27 run(s), 497 cell(s).
Incomplete runs (no end record): ecf0fbc9-fa5d-46a6-958a-4e5219356f3b.

Items/s = items completed / cell wall time (all callers). Latency is per call. Padding efficiency = real tokens / padded cells (rows x longest row per ORT run, as the library chunks). Quality: every output vs the same item run alone, worst difference against the profile's padded-batch tolerance. Micro-batched configurations (micro-wW-rR: MaxWait W ms, MaxRows R) issue every item as its own call through the library's batcher; their padding is the dispatched batches', and 'micro rows / queue ms' is the mean rows per dispatched batch and the mean time an item waited in the queue.

## embeddinggemma-2 · cuda · baseline · intra-op 4 · max_batch_size 8 · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 62.1 | 74.9 | 73.5 |
| 32 | 62.4 | 73.9 | 74.6 |
| 64 | 62.1 | 74.6 | 74.5 |

Best: batch 16 at concurrency 4, 74.9 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 10 | 62.1 | 258.24 | n/a | 0.906 | tokenize 9.95, encode 247.57 | 2407 | 20020 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 74.9 | 849.44 | n/a | 0.908 | tokenize 11.45, encode 834.37 | 2408 | 20044 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 16 | 73.5 | 3458.21 | n/a | 0.907 | tokenize 36.68, encode 3420.52 | 2409 | 20140 |  | 1.96e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 62.4 | 514.52 | n/a | 0.906 | tokenize 18.85, encode 494.67 | 2405 | 20020 |  | 1.96e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 73.9 | 1698.98 | n/a | 0.906 | tokenize 25.15, encode 1672.32 | 2408 | 20044 |  | 1.96e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 16 | 74.6 | 6530.97 | n/a | 0.907 | tokenize 75.63, encode 6452.87 | 2425 | 20140 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 62.1 | 1030.22 | n/a | 0.907 | tokenize 37.73, encode 990.15 | 2338 | 20028 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 74.6 | 3396.12 | n/a | 0.907 | tokenize 40.43, encode 3339.07 | 2120 | 20044 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 74.5 | 13732.89 | n/a | 0.907 | tokenize 210.32, encode 13513.24 | 2105 | 20140 |  | 1.96e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 72.1 | 84.0 | 81.4 |
| 32 | 71.5 | 81.8 | 82.2 |
| 64 | 70.4 | 80.2 | 80.8 |

Best: batch 16 at concurrency 4, 84.0 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 10 | 72.1 | 233.22 | n/a | 0.388 | tokenize 6.02, encode 209.40 | 1258 | 3424 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 84.0 | 753.57 | n/a | 0.391 | tokenize 4.76, encode 723.30 | 1424 | 5548 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 19 | 81.4 | 3146.79 | n/a | 0.394 | tokenize 18.99, encode 3126.14 | 1965 | 13996 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 71.5 | 434.85 | n/a | 0.405 | tokenize 8.35, encode 413.75 | 1912 | 13876 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 81.8 | 1499.11 | n/a | 0.405 | tokenize 8.65, encode 1490.66 | 1965 | 13900 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 16 | 82.2 | 6031.79 | n/a | 0.402 | tokenize 31.26, encode 5999.26 | 2186 | 19116 |  | 3.57e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 70.4 | 1017.62 | n/a | 0.391 | tokenize 15.37, encode 890.62 | 2189 | 18996 |  | 3.50e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 80.2 | 3078.32 | n/a | 0.391 | tokenize 17.30, encode 3058.04 | 2318 | 19020 |  | 3.50e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 80.8 | 12638.69 | n/a | 0.391 | tokenize 74.09, encode 12562.64 | 2406 | 20140 |  | 3.50e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · baseline · intra-op 4 · max_batch_size call (1) · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 49.3 | 67.2 | 69.5 |

Best: batch 1 at concurrency 16, 69.5 items/s; best single-item callers: concurrency 16, 69.5 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 99 | 49.3 | 20.34 | 21.63 | 1.000 | tokenize 0.69, encode 19.72 | 1332 | 2836 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 137 | 67.2 | 59.24 | 63.07 | 1.000 | tokenize 0.81, encode 58.27 | 1333 | 2828 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 148 | 69.5 | 223.85 | 308.58 | 1.000 | tokenize 1.16, encode 222.56 | 1355 | 3948 |  | 0.00e+00 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 64.9 | 131.2 | 142.3 |

Best: batch 1 at concurrency 16, 142.3 items/s; best single-item callers: concurrency 16, 142.3 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 130 | 64.9 | 14.07 | 20.54 | 1.000 | tokenize 0.16, encode 13.96 | 1249 | 1780 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 265 | 131.2 | 28.75 | 47.93 | 1.000 | tokenize 0.17, encode 28.28 | 1297 | 2828 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 294 | 142.3 | 111.30 | 149.84 | 1.000 | tokenize 0.23, encode 111.21 | 1331 | 2924 |  | 0.00e+00 (5e-06) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 72.2 | 177.3 | 200.8 |

Best: batch 1 at concurrency 16, 200.8 items/s; best single-item callers: concurrency 16, 200.8 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 145 | 72.2 | 13.74 | 14.71 | 1.000 | tokenize 0.04, encode 13.71 | 1226 | 1568 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 358 | 177.3 | 22.55 | 24.78 | 1.000 | tokenize 0.04, encode 22.51 | 1232 | 1644 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 413 | 200.8 | 69.26 | 126.94 | 1.000 | tokenize 0.05, encode 69.21 | 1248 | 1900 |  | 0.00e+00 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · baseline · intra-op 4 · max_batch_size call (64) · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 48.4 | 67.4 | 69.3 |
| 4 | 59.4 | 72.5 | 72.1 |
| 8 | 61.8 | 75.2 | 74.3 |
| 16 | 64.0 | 70.8 | 68.5 |
| 32 | 63.7 | 71.6 |  |

Best: batch 8 at concurrency 4, 75.2 items/s; best single-item callers: concurrency 16, 69.3 items/s (ratio 1.09).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 97 | 48.4 | 20.70 | 22.55 | 1.000 | tokenize 0.92, encode 19.61 | 1625 | 8036 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 136 | 67.4 | 59.17 | 62.80 | 1.000 | tokenize 0.70, encode 58.47 | 1625 | 7948 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 149 | 69.3 | 223.32 | 277.20 | 1.000 | tokenize 1.29, encode 222.13 | 1638 | 8044 |  | 0.00e+00 (5e-06) | 0 | pass |
| 4 | 1 | 1 | 30 | 59.4 | 68.00 | 69.36 | 0.921 | tokenize 2.95, encode 64.98 | 1638 | 7924 |  | 2.01e-07 (5e-06) | 0 | pass |
| 4 | 4 | 1 | 40 | 72.5 | 219.29 | 232.76 | 0.920 | tokenize 3.16, encode 216.61 | 1662 | 7948 |  | 2.01e-07 (5e-06) | 0 | pass |
| 4 | 16 | 1 | 45 | 72.1 | 779.38 | 1255.72 | 0.918 | tokenize 6.60, encode 776.44 | 1708 | 9068 |  | 2.01e-07 (5e-06) | 0 | pass |
| 8 | 1 | 1 | 16 | 61.8 | 129.48 | n/a | 0.907 | tokenize 6.26, encode 123.25 | 1714 | 9060 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 20 | 75.2 | 420.94 | 447.56 | 0.907 | tokenize 6.47, encode 413.72 | 1685 | 8972 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 27 | 74.3 | 1655.54 | 2892.93 | 0.909 | tokenize 17.91, encode 1639.38 | 1915 | 14188 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 64.0 | 250.64 | n/a | 0.900 | tokenize 12.57, encode 237.92 | 1939 | 14180 |  | 2.60e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 70.8 | 896.58 | n/a | 0.901 | tokenize 14.80, encode 881.71 | 2000 | 14092 |  | 2.60e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 16 | 68.5 | 2337.31 | n/a | 0.900 | tokenize 42.11, encode 2299.61 | 2285 | 14188 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 63.7 | 497.65 | n/a | 0.900 | tokenize 19.79, encode 476.21 | 2282 | 14108 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 71.6 | 1774.88 | n/a | 0.900 | tokenize 23.96, encode 1748.59 | 2379 | 14092 |  | 2.60e-07 (5e-06) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 75.7 | 173.4 | 202.6 |
| 4 | 308.1 | 721.6 | 734.1 |
| 8 | 589.5 | 1005.5 | 1078.0 |
| 16 | 872.3 | 1171.5 | 1195.0 |
| 32 | 1017.4 | 1254.2 | 1254.2 |
| 64 | 1070.7 | 1279.1 | 1268.9 |

Best: batch 64 at concurrency 4, 1279.1 items/s; best single-item callers: concurrency 16, 202.6 items/s (ratio 6.32).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 152 | 75.7 | 13.10 | 13.60 | 1.000 | tokenize 0.08, encode 13.00 | 1230 | 1568 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 350 | 173.4 | 22.47 | 24.88 | 1.000 | tokenize 0.04, encode 22.42 | 1236 | 1644 |  | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 415 | 202.6 | 68.59 | 124.54 | 1.000 | tokenize 0.05, encode 68.55 | 1251 | 1900 |  | 0.00e+00 (5e-06) | 0 | pass |
| 4 | 1 | 1 | 155 | 308.1 | 12.95 | 13.43 | 0.814 | tokenize 0.10, encode 12.82 | 1252 | 1796 |  | 1.74e-07 (5e-06) | 0 | pass |
| 4 | 4 | 1 | 364 | 721.6 | 21.78 | 24.12 | 0.815 | tokenize 0.12, encode 21.62 | 1255 | 1804 |  | 1.74e-07 (5e-06) | 0 | pass |
| 4 | 16 | 1 | 376 | 734.1 | 78.65 | 124.81 | 0.815 | tokenize 0.17, encode 78.42 | 1289 | 2924 |  | 1.74e-07 (5e-06) | 0 | pass |
| 8 | 1 | 1 | 148 | 589.5 | 13.59 | 14.08 | 0.757 | tokenize 0.21, encode 13.38 | 1289 | 2812 |  | 1.13e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 254 | 1005.5 | 31.88 | 32.90 | 0.757 | tokenize 0.21, encode 31.59 | 1294 | 2828 |  | 1.13e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 280 | 1078.0 | 115.56 | 141.19 | 0.757 | tokenize 0.38, encode 115.13 | 1310 | 2924 |  | 1.13e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 110 | 872.3 | 18.35 | 19.23 | 0.699 | tokenize 0.43, encode 17.95 | 1310 | 2804 |  | 2.15e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 148 | 1171.5 | 54.39 | 56.90 | 0.699 | tokenize 0.41, encode 53.86 | 1310 | 2828 |  | 2.15e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 162 | 1195.0 | 205.79 | 260.70 | 0.700 | tokenize 0.80, encode 204.32 | 1347 | 3948 |  | 2.15e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 64 | 1017.4 | 30.93 | 33.27 | 0.654 | tokenize 0.82, encode 30.14 | 1346 | 3940 |  | 1.64e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 80 | 1254.2 | 101.29 | 105.84 | 0.654 | tokenize 0.82, encode 100.21 | 1357 | 3852 |  | 1.64e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 87 | 1254.2 | 384.78 | 586.28 | 0.654 | tokenize 1.43, encode 384.07 | 1477 | 5996 |  | 1.64e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 34 | 1070.7 | 61.81 | 62.60 | 0.644 | tokenize 1.71, encode 57.94 | 1490 | 5988 |  | 1.86e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 41 | 1279.1 | 199.43 | 211.62 | 0.644 | tokenize 1.78, encode 197.79 | 1522 | 5900 |  | 1.86e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 47 | 1268.9 | 776.01 | 1077.16 | 0.644 | tokenize 3.64, encode 773.72 | 1625 | 8044 |  | 1.86e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · baseline · intra-op 4 · max_batch_size default · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 61.8 |
| 16 | 64.7 |
| 32 | 64.3 |
| 64 | 64.7 |

Best: batch 64 at concurrency 1, 64.7 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 16 | 61.8 | 128.89 | n/a | 0.907 | tokenize 5.36, encode 123.19 | 1262 | 3490 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 64.7 | 247.58 | n/a | 0.900 | tokenize 10.08, encode 237.29 | 1262 | 3490 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 64.3 | 495.54 | n/a | 0.900 | tokenize 20.09, encode 474.96 | 1262 | 3490 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 64.7 | 991.95 | n/a | 0.900 | tokenize 36.51, encode 952.22 | 1262 | 3490 |  | 2.60e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 72.6 |
| 16 | 70.0 |
| 32 | 68.3 |
| 64 | 69.3 |

Best: batch 8 at concurrency 1, 72.6 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 19 | 72.6 | 106.38 | n/a | 0.416 | tokenize 2.20, encode 105.10 | 1258 | 3490 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 70.0 | 221.90 | n/a | 0.359 | tokenize 4.44, encode 216.74 | 1261 | 3490 |  | 2.60e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 68.3 | 441.09 | n/a | 0.370 | tokenize 8.87, encode 429.48 | 1262 | 3490 |  | 2.60e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 69.3 | 982.38 | n/a | 0.364 | tokenize 14.92, encode 907.59 | 1262 | 3490 |  | 2.60e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · micro-w0-r8 · intra-op 4 · max_batch_size call (8) · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 47.8 | 56.8 | 64.4 |

Best: batch 1 at concurrency 16, 64.4 items/s; best single-item callers: concurrency 16, 64.4 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 96 | 47.8 | 20.85 | 22.59 | 1.000 | call 20.82 | 1405 | 3370 | 1.00 / 0.02 | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 117 | 56.8 | 70.23 | 72.95 | 0.949 | call 70.23 | 1405 | 3370 | 1.98 / 26.43 | 2.11e-07 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 137 | 64.4 | 248.14 | 260.96 | 0.909 | call 248.14 | 1407 | 3370 | 7.61 / 115.68 | 2.62e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 63.1 | 74.9 | 66.6 |

Best: batch 1 at concurrency 4, 74.9 items/s; best single-item callers: concurrency 4, 74.9 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 127 | 63.1 | 14.28 | 21.56 | 1.000 | call 14.28 | 1341 | 2346 | 1.00 / 0.03 | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 153 | 74.9 | 52.86 | 129.77 | 0.586 | call 52.86 | 1348 | 2346 | 1.99 / 20.36 | 2.11e-07 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 145 | 66.6 | 227.52 | 407.11 | 0.350 | call 227.52 | 1401 | 3370 | 7.63 / 112.61 | 3.55e-07 (5e-06) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 70.3 | 141.2 | 575.9 |

Best: batch 1 at concurrency 16, 575.9 items/s; best single-item callers: concurrency 16, 575.9 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 141 | 70.3 | 14.13 | 14.54 | 1.000 | call 14.13 | 1336 | 2346 | 1.00 / 0.02 | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 285 | 141.2 | 27.91 | 28.63 | 0.889 | call 27.91 | 1337 | 2346 | 1.99 / 13.83 | 1.29e-07 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 1161 | 575.9 | 27.53 | 28.13 | 0.750 | call 27.53 | 1341 | 2346 | 7.95 / 13.57 | 1.13e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · micro-w2-r16 · intra-op 4 · max_batch_size call (16) · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 44.0 | 59.4 | 65.4 |

Best: batch 1 at concurrency 16, 65.4 items/s; best single-item callers: concurrency 16, 65.4 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 88 | 44.0 | 22.79 | 24.24 | 1.000 | call 22.75 | 1525 | 3498 | 1.00 / 2.00 | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 120 | 59.4 | 67.68 | 70.90 | 0.923 | call 67.60 | 1534 | 3498 | 4.00 / 1.56 | 2.01e-07 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 144 | 65.4 | 245.18 | 251.98 | 0.909 | call 245.17 | 1530 | 3498 | 8.00 / 85.67 | 2.60e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 58.2 | 77.5 | 66.8 |

Best: batch 1 at concurrency 4, 77.5 items/s; best single-item callers: concurrency 4, 77.5 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 117 | 58.2 | 15.36 | 23.22 | 1.000 | call 15.36 | 1455 | 2474 | 1.00 / 2.00 | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 159 | 77.5 | 53.76 | 71.50 | 0.587 | call 53.76 | 1465 | 2474 | 2.04 / 19.57 | 2.11e-07 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 144 | 66.8 | 212.59 | 355.61 | 0.364 | call 212.57 | 1525 | 3498 | 8.00 / 104.02 | 3.57e-07 (5e-06) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 64.2 | 166.7 | 557.2 |

Best: batch 1 at concurrency 16, 557.2 items/s; best single-item callers: concurrency 16, 557.2 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 129 | 64.2 | 15.26 | 16.30 | 1.000 | call 15.26 | 1449 | 2474 | 1.00 / 2.00 | 0.00e+00 (5e-06) | 0 | pass |
| 1 | 4 | 1 | 336 | 166.7 | 26.11 | 26.77 | 0.877 | call 26.11 | 1450 | 2474 | 2.26 / 10.44 | 1.74e-07 (5e-06) | 0 | pass |
| 1 | 16 | 1 | 1125 | 557.2 | 28.96 | 30.34 | 0.737 | call 28.96 | 1455 | 2474 | 7.98 / 13.50 | 1.13e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · sorted · intra-op 4 · max_batch_size 8 · max_batch_tokens 8192 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 63.9 | 78.0 | 76.4 |
| 32 | 65.6 | 78.1 | 78.1 |
| 64 | 67.0 | 79.4 | 79.3 |

Best: batch 64 at concurrency 4, 79.4 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 10 | 63.9 | 248.48 | n/a | 0.945 | tokenize 12.16, encode 234.80 | 2333 | 15998 |  | 2.69e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 78.0 | 819.85 | n/a | 0.945 | tokenize 12.59, encode 805.32 | 2338 | 16014 |  | 2.69e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 16 | 76.4 | 3106.26 | n/a | 0.945 | tokenize 54.04, encode 3055.45 | 2389 | 16110 |  | 2.69e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 65.6 | 487.87 | n/a | 0.970 | tokenize 20.29, encode 466.44 | 2385 | 15990 |  | 2.69e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 78.1 | 1599.11 | n/a | 0.970 | tokenize 21.82, encode 1579.08 | 2318 | 16014 |  | 2.69e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 16 | 78.1 | 6328.84 | n/a | 0.970 | tokenize 94.78, encode 6239.76 | 2384 | 19182 |  | 2.69e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 67.0 | 953.44 | n/a | 0.987 | tokenize 37.29, encode 916.00 | 2387 | 19070 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 79.4 | 3190.09 | n/a | 0.987 | tokenize 41.92, encode 3136.64 | 2391 | 19086 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 79.3 | 12157.15 | n/a | 0.987 | tokenize 186.03, encode 11991.93 | 2476 | 22254 |  | 1.96e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 102.6 | 118.3 | 119.5 |
| 32 | 123.6 | 150.7 | 151.2 |
| 64 | 142.2 | 171.9 | 174.9 |

Best: batch 64 at concurrency 16, 174.9 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 13 | 102.6 | 154.25 | n/a | 0.575 | tokenize 4.59, encode 147.32 | 1229 | 2466 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 17 | 118.3 | 507.99 | n/a | 0.579 | tokenize 4.67, encode 503.63 | 1404 | 5612 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 25 | 119.5 | 1877.07 | 2342.07 | 0.581 | tokenize 11.63, encode 1869.86 | 1790 | 12014 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 123.6 | 256.91 | n/a | 0.730 | tokenize 8.24, encode 243.71 | 1797 | 11894 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 12 | 150.7 | 811.27 | n/a | 0.728 | tokenize 9.00, encode 788.62 | 1893 | 11918 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 17 | 151.2 | 3116.34 | n/a | 0.728 | tokenize 26.66, encode 3087.78 | 2263 | 15086 |  | 3.57e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 142.2 | 477.73 | n/a | 0.850 | tokenize 14.83, encode 435.39 | 2264 | 14966 |  | 3.57e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 171.9 | 1390.97 | n/a | 0.850 | tokenize 21.41, encode 1344.87 | 2228 | 14990 |  | 3.57e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 174.9 | 5764.22 | n/a | 0.850 | tokenize 78.10, encode 5680.04 | 2332 | 16110 |  | 3.57e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · sorted+tokens4096 · intra-op 4 · max_batch_size call (64) · max_batch_tokens 4096 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 62.0 | 74.1 | 75.1 |
| 16 | 63.6 | 77.2 | 77.6 |
| 32 | 65.7 | 77.9 | 77.7 |
| 64 | 66.3 | 79.7 | 80.0 |

Best: batch 64 at concurrency 16, 80.0 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 16 | 62.0 | 129.36 | n/a | 0.907 | tokenize 4.94, encode 123.97 | 2051 | 14972 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 20 | 74.1 | 431.75 | 436.99 | 0.907 | tokenize 6.26, encode 423.39 | 2038 | 14988 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 31 | 75.1 | 1686.20 | 1689.63 | 0.907 | tokenize 16.28, encode 1663.81 | 1949 | 15084 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 63.6 | 251.23 | n/a | 0.945 | tokenize 11.72, encode 235.28 | 1937 | 15076 |  | 2.69e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 77.2 | 823.23 | n/a | 0.945 | tokenize 13.45, encode 807.62 | 1928 | 14988 |  | 2.69e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 18 | 77.6 | 3386.91 | n/a | 0.945 | tokenize 38.18, encode 3343.29 | 1922 | 15084 |  | 2.69e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 65.7 | 487.68 | n/a | 0.972 | tokenize 20.69, encode 461.38 | 1922 | 14972 |  | 2.66e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 77.9 | 1618.14 | n/a | 0.972 | tokenize 30.65, encode 1577.61 | 1910 | 14988 |  | 2.66e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 16 | 77.7 | 6378.21 | n/a | 0.972 | tokenize 85.11, encode 6292.92 | 1984 | 17132 |  | 2.66e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 66.3 | 967.16 | n/a | 0.983 | tokenize 50.00, encode 916.23 | 1992 | 17012 |  | 2.62e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 79.7 | 3174.68 | n/a | 0.983 | tokenize 45.78, encode 3128.41 | 2006 | 17036 |  | 2.62e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 80.0 | 12658.57 | n/a | 0.983 | tokenize 169.27, encode 12504.33 | 2173 | 20204 |  | 2.62e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 77.5 | 89.5 | 87.3 |
| 16 | 102.7 | 118.4 | 117.4 |
| 32 | 110.5 | 130.0 | 129.1 |
| 64 | 129.9 | 152.3 | 153.1 |

Best: batch 64 at concurrency 16, 153.1 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 20 | 77.5 | 106.85 | 129.45 | 0.437 | tokenize 2.07, encode 105.47 | 1237 | 2464 |  | 3.57e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 25 | 89.5 | 313.52 | 591.16 | 0.429 | tokenize 2.06, encode 311.39 | 1386 | 4588 |  | 3.57e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 32 | 87.3 | 1382.19 | 2847.56 | 0.438 | tokenize 4.57, encode 1372.25 | 1782 | 12012 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 13 | 102.7 | 155.13 | n/a | 0.555 | tokenize 4.27, encode 150.68 | 1783 | 11892 |  | 2.58e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 16 | 118.4 | 525.46 | n/a | 0.565 | tokenize 5.29, encode 516.16 | 1839 | 11916 |  | 2.58e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 25 | 117.4 | 1888.67 | 2902.54 | 0.561 | tokenize 9.70, encode 1884.49 | 1939 | 12012 |  | 2.58e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 110.5 | 284.20 | n/a | 0.630 | tokenize 7.96, encode 269.93 | 1939 | 11892 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 12 | 130.0 | 982.13 | n/a | 0.627 | tokenize 9.23, encode 969.40 | 1947 | 11916 |  | 3.57e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 16 | 129.1 | 3786.48 | n/a | 0.627 | tokenize 34.04, encode 3752.65 | 1957 | 13036 |  | 3.57e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 129.9 | 529.02 | n/a | 0.712 | tokenize 15.18, encode 477.27 | 1953 | 12916 |  | 3.50e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 152.3 | 1551.52 | n/a | 0.712 | tokenize 18.13, encode 1522.12 | 1971 | 12940 |  | 3.50e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 153.1 | 6012.76 | n/a | 0.712 | tokenize 60.27, encode 5952.95 | 2050 | 15084 |  | 3.50e-07 (5e-06) | 0 | pass |


## embeddinggemma-2 · cuda · tokens4096 · intra-op 4 · max_batch_size call (64) · max_batch_tokens 4096 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 60.9 | 74.1 | 73.4 |
| 16 | 60.0 | 75.2 | 74.0 |
| 32 | 61.3 | 74.4 | 74.2 |
| 64 | 62.2 | 74.4 | 74.5 |

Best: batch 16 at concurrency 4, 75.2 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 16 | 60.9 | 130.76 | n/a | 0.907 | tokenize 7.01, encode 123.71 | 2214 | 19076 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 20 | 74.1 | 421.41 | 473.18 | 0.907 | tokenize 6.53, encode 416.30 | 2250 | 19084 |  | 1.96e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 30 | 73.4 | 1677.56 | 3257.97 | 0.908 | tokenize 22.97, encode 1612.46 | 2422 | 19180 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 60.0 | 265.24 | n/a | 0.906 | tokenize 14.31, encode 247.52 | 2422 | 19172 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 75.2 | 841.24 | n/a | 0.908 | tokenize 11.13, encode 829.04 | 2422 | 19084 |  | 1.96e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 16 | 74.0 | 3026.74 | n/a | 0.907 | tokenize 46.66, encode 2980.78 | 2343 | 19180 |  | 1.96e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 61.3 | 520.29 | n/a | 0.906 | tokenize 21.19, encode 495.57 | 2298 | 19076 |  | 1.96e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 74.4 | 1704.20 | n/a | 0.906 | tokenize 26.52, encode 1672.52 | 2303 | 19084 |  | 1.96e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 16 | 74.2 | 6768.24 | n/a | 0.907 | tokenize 105.14, encode 6661.61 | 2306 | 19180 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 62.2 | 1030.44 | n/a | 0.907 | tokenize 37.54, encode 990.86 | 2306 | 19060 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 74.4 | 3411.46 | n/a | 0.907 | tokenize 49.07, encode 3359.45 | 2307 | 19084 |  | 1.96e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 74.5 | 13008.45 | n/a | 0.907 | tokenize 220.22, encode 12693.91 | 2225 | 20204 |  | 1.96e-07 (5e-06) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 76.0 | 87.7 | 87.0 |
| 16 | 75.9 | 87.9 | 85.2 |
| 32 | 71.6 | 82.9 | 83.8 |
| 64 | 72.2 | 83.5 | 84.1 |

Best: batch 16 at concurrency 4, 87.9 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 20 | 76.0 | 106.92 | 129.90 | 0.429 | tokenize 1.99, encode 105.59 | 1236 | 2464 |  | 3.57e-07 (5e-06) | 0 | pass |
| 8 | 4 | 1 | 25 | 87.7 | 322.17 | 616.07 | 0.421 | tokenize 2.12, encode 320.78 | 1391 | 4588 |  | 3.57e-07 (5e-06) | 0 | pass |
| 8 | 16 | 1 | 30 | 87.0 | 1366.90 | 2692.01 | 0.427 | tokenize 4.83, encode 1358.47 | 1841 | 13036 |  | 3.57e-07 (5e-06) | 0 | pass |
| 16 | 1 | 1 | 10 | 75.9 | 224.47 | n/a | 0.407 | tokenize 4.18, encode 216.98 | 1844 | 12916 |  | 2.55e-07 (5e-06) | 0 | pass |
| 16 | 4 | 1 | 12 | 87.9 | 687.06 | n/a | 0.407 | tokenize 4.92, encode 674.33 | 1924 | 12940 |  | 2.55e-07 (5e-06) | 0 | pass |
| 16 | 16 | 1 | 17 | 85.2 | 2814.89 | n/a | 0.416 | tokenize 15.57, encode 2800.16 | 1939 | 13036 |  | 2.55e-07 (5e-06) | 0 | pass |
| 32 | 1 | 1 | 10 | 71.6 | 460.95 | n/a | 0.408 | tokenize 11.71, encode 429.86 | 1939 | 12916 |  | 2.55e-07 (5e-06) | 0 | pass |
| 32 | 4 | 1 | 10 | 82.9 | 1473.53 | n/a | 0.408 | tokenize 10.41, encode 1444.14 | 1943 | 12940 |  | 2.55e-07 (5e-06) | 0 | pass |
| 32 | 16 | 1 | 16 | 83.8 | 5690.82 | n/a | 0.402 | tokenize 28.63, encode 5661.92 | 2005 | 15084 |  | 2.55e-07 (5e-06) | 0 | pass |
| 64 | 1 | 1 | 10 | 72.2 | 936.49 | n/a | 0.400 | tokenize 16.49, encode 866.02 | 2006 | 14964 |  | 3.55e-07 (5e-06) | 0 | pass |
| 64 | 4 | 1 | 10 | 83.5 | 2818.29 | n/a | 0.400 | tokenize 19.39, encode 2798.04 | 2058 | 14988 |  | 3.55e-07 (5e-06) | 0 | pass |
| 64 | 16 | 1 | 16 | 84.1 | 11899.52 | n/a | 0.400 | tokenize 83.24, encode 11749.27 | 2212 | 19180 |  | 3.55e-07 (5e-06) | 0 | pass |


## gliner2-base-v1 · cuda · baseline · intra-op 4 · max_batch_size 8 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 70.8 | 71.7 | 71.7 |
| 32 | 70.9 | 72.2 | 71.8 |
| 64 | 70.6 | 72.3 | 71.8 |

Best: batch 64 at concurrency 4, 72.3 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 10 | 70.8 | 226.15 | n/a | 0.965 | prepare 2.05, inference 220.03, decode 1.75 | 1164 | 4498 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 12 | 71.7 | 680.90 | n/a | 0.967 | prepare 1.87, inference 607.95, decode 1.78 | 1169 | 4520 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 22 | 71.7 | 3442.61 | 4678.30 | 0.967 | prepare 4.72, inference 3435.94, decode 1.82 | 1188 | 4612 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 70.9 | 451.56 | n/a | 0.967 | prepare 3.88, inference 440.16, decode 3.60 | 1196 | 4500 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 72.2 | 1418.83 | n/a | 0.967 | prepare 7.34, inference 1369.14, decode 3.56 | 1195 | 4522 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 16 | 71.8 | 5470.78 | n/a | 0.967 | prepare 58.47, inference 5364.20, decode 3.89 | 1197 | 4604 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 70.6 | 903.94 | n/a | 0.967 | prepare 7.42, inference 877.35, decode 7.08 | 1177 | 4500 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 72.3 | 3077.46 | n/a | 0.967 | prepare 12.25, inference 2799.55, decode 7.31 | 1194 | 4522 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 16 | 71.8 | 11491.48 | n/a | 0.967 | prepare 123.65, inference 11238.63, decode 7.20 | 1243 | 4622 |  | 1.79e-05 (1e-04) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 110.4 | 109.9 | 108.5 |
| 32 | 107.2 | 107.8 | 108.5 |
| 64 | 107.8 | 108.9 | 108.4 |

Best: batch 16 at concurrency 1, 110.4 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 14 | 110.4 | 132.69 | n/a | 0.400 | prepare 0.55, inference 127.80, decode 0.44 | 1143 | 4450 |  | 1.99e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 17 | 109.9 | 450.01 | n/a | 0.403 | prepare 0.64, inference 448.42, decode 0.54 | 1148 | 4492 |  | 1.99e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 28 | 108.5 | 1346.11 | 3428.52 | 0.397 | prepare 0.75, inference 1317.22, decode 0.53 | 1159 | 4592 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 107.2 | 327.37 | n/a | 0.409 | prepare 1.13, inference 290.46, decode 0.99 | 1161 | 4496 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 107.8 | 1005.42 | n/a | 0.409 | prepare 1.25, inference 952.92, decode 1.03 | 1161 | 4518 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 19 | 108.5 | 2859.64 | n/a | 0.408 | prepare 9.66, inference 2849.01, decode 1.01 | 1169 | 4584 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 107.8 | 591.12 | n/a | 0.404 | prepare 2.63, inference 585.51, decode 2.03 | 1166 | 4496 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 108.9 | 2066.28 | n/a | 0.404 | prepare 5.71, inference 2015.17, decode 2.09 | 1171 | 4518 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 17 | 108.4 | 7257.92 | n/a | 0.403 | prepare 20.35, inference 7227.42, decode 2.12 | 1180 | 4602 |  | 1.99e-05 (1e-04) | 0 | pass |


## gliner2-base-v1 · cuda · baseline · intra-op 4 · max_batch_size call (1) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 61.0 | 60.4 | 59.5 |

Best: batch 1 at concurrency 1, 61.0 items/s; best single-item callers: concurrency 1, 61.0 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 122 | 61.0 | 16.27 | 17.28 | 1.000 | prepare 0.08, inference 16.05, decode 0.11 | 1159 | 1500 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 4 | 1 | 124 | 60.4 | 65.62 | 68.48 | 1.000 | prepare 0.18, inference 65.04, decode 0.25 | 1159 | 1524 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 16 | 1 | 135 | 59.5 | 266.45 | 275.95 | 1.000 | prepare 0.16, inference 265.83, decode 0.23 | 1159 | 1622 |  | 0.00e+00 (1e-04) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 145.9 | 106.4 | 103.8 |

Best: batch 1 at concurrency 1, 145.9 items/s; best single-item callers: concurrency 1, 145.9 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 292 | 145.9 | 4.77 | 16.12 | 1.000 | prepare 0.02, inference 4.71, decode 0.02 | 1153 | 1502 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 4 | 1 | 217 | 106.4 | 36.28 | 48.17 | 1.000 | prepare 0.03, inference 36.10, decode 0.04 | 1158 | 1524 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 16 | 1 | 225 | 103.8 | 153.02 | 177.49 | 1.000 | prepare 0.03, inference 152.72, decode 0.04 | 1159 | 1622 |  | 0.00e+00 (1e-04) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 216.1 | 119.1 | 122.1 |

Best: batch 1 at concurrency 1, 216.1 items/s; best single-item callers: concurrency 1, 216.1 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 433 | 216.1 | 4.41 | 5.50 | 1.000 | prepare 0.01, inference 4.38, decode 0.01 | 1138 | 1711 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 4 | 1 | 244 | 119.1 | 33.29 | 38.17 | 1.000 | prepare 0.02, inference 33.24, decode 0.03 | 1145 | 1753 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 16 | 1 | 262 | 122.1 | 132.06 | 144.24 | 1.000 | prepare 0.02, inference 131.99, decode 0.02 | 1152 | 1622 |  | 0.00e+00 (1e-04) | 0 | pass |


## gliner2-base-v1 · cuda · baseline · intra-op 4 · max_batch_size call (64) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 60.7 | 60.1 | 59.4 |
| 4 | 70.5 | 71.0 | 70.2 |
| 8 | 71.1 | 72.3 | 71.8 |
| 16 | 73.4 | 74.4 | 74.0 |
| 32 | 74.0 | 75.2 | 74.9 |
| 64 | 74.7 | 76.4 | 76.3 |

Best: batch 64 at concurrency 4, 76.4 items/s; best single-item callers: concurrency 1, 60.7 items/s (ratio 1.26).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 122 | 60.7 | 16.35 | 17.41 | 1.000 | prepare 0.09, inference 16.09, decode 0.11 | 1166 | 3540 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 4 | 1 | 124 | 60.1 | 66.05 | 68.47 | 1.000 | prepare 0.18, inference 65.42, decode 0.25 | 1166 | 3564 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 16 | 1 | 135 | 59.4 | 266.42 | 275.74 | 1.000 | prepare 0.18, inference 265.75, decode 0.23 | 1166 | 3662 |  | 0.00e+00 (1e-04) | 0 | pass |
| 4 | 1 | 1 | 36 | 70.5 | 56.78 | 58.62 | 0.972 | prepare 0.40, inference 55.73, decode 0.42 | 1167 | 3542 |  | 1.79e-05 (1e-04) | 0 | pass |
| 4 | 4 | 1 | 39 | 71.0 | 223.55 | 229.37 | 0.972 | prepare 0.47, inference 222.46, decode 0.46 | 1167 | 3564 |  | 1.79e-05 (1e-04) | 0 | pass |
| 4 | 16 | 1 | 51 | 70.2 | 897.04 | 924.56 | 0.972 | prepare 0.63, inference 896.11, decode 0.50 | 1170 | 3662 |  | 1.79e-05 (1e-04) | 0 | pass |
| 8 | 1 | 1 | 18 | 71.1 | 112.72 | n/a | 0.968 | prepare 0.84, inference 110.82, decode 0.86 | 1173 | 4556 |  | 1.79e-05 (1e-04) | 0 | pass |
| 8 | 4 | 1 | 22 | 72.3 | 438.78 | 446.72 | 0.967 | prepare 1.00, inference 435.57, decode 0.87 | 1165 | 4588 |  | 1.79e-05 (1e-04) | 0 | pass |
| 8 | 16 | 1 | 33 | 71.8 | 1764.80 | 1789.15 | 0.967 | prepare 1.12, inference 1762.25, decode 0.89 | 1172 | 4686 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 10 | 73.4 | 215.82 | n/a | 0.963 | prepare 1.72, inference 212.27, decode 1.72 | 1171 | 4564 |  | 2.95e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 13 | 74.4 | 853.88 | n/a | 0.965 | prepare 3.31, inference 848.88, decode 1.79 | 1172 | 4588 |  | 2.95e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 25 | 74.0 | 2813.05 | 3460.11 | 0.965 | prepare 7.30, inference 2790.17, decode 1.86 | 1187 | 4646 |  | 2.95e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 74.0 | 428.20 | n/a | 0.964 | prepare 3.35, inference 421.08, decode 3.50 | 1292 | 10700 |  | 2.32e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 75.2 | 1685.57 | n/a | 0.964 | prepare 8.29, inference 1674.67, decode 3.62 | 1305 | 10732 |  | 2.32e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 20 | 74.9 | 4716.03 | 6796.47 | 0.963 | prepare 55.59, inference 4416.25, decode 3.67 | 1316 | 10750 |  | 2.32e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 74.7 | 847.08 | n/a | 0.963 | prepare 7.41, inference 831.13, decode 6.99 | 1593 | 20940 |  | 2.31e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 76.4 | 3315.88 | n/a | 0.963 | prepare 8.84, inference 3299.69, decode 7.12 | 1604 | 20972 |  | 2.31e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 18 | 76.3 | 8390.53 | n/a | 0.963 | prepare 129.07, inference 7828.48, decode 7.48 | 1621 | 20972 |  | 2.31e-05 (1e-04) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 147.3 | 104.8 | 105.6 |
| 4 | 125.5 | 123.9 | 121.5 |
| 8 | 110.8 | 110.1 | 108.4 |
| 16 | 84.4 | 83.9 | 83.8 |
| 32 | 82.2 | 82.3 | 82.7 |
| 64 | 81.3 | 81.7 | 81.7 |

Best: batch 1 at concurrency 1, 147.3 items/s; best single-item callers: concurrency 1, 147.3 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 295 | 147.3 | 4.73 | 16.17 | 1.000 | prepare 0.02, inference 4.67, decode 0.02 | 1621 | 20948 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 4 | 1 | 214 | 104.8 | 37.69 | 49.46 | 1.000 | prepare 0.04, inference 37.40, decode 0.04 | 1621 | 20972 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 16 | 1 | 229 | 105.6 | 149.40 | 175.09 | 1.000 | prepare 0.04, inference 149.30, decode 0.05 | 1621 | 21070 |  | 0.00e+00 (1e-04) | 0 | pass |
| 4 | 1 | 1 | 63 | 125.5 | 23.48 | 57.13 | 0.492 | prepare 0.13, inference 23.25, decode 0.11 | 1621 | 20948 |  | 2.77e-05 (1e-04) | 0 | pass |
| 4 | 4 | 1 | 66 | 123.9 | 124.74 | 180.24 | 0.503 | prepare 0.17, inference 124.34, decode 0.14 | 1586 | 20972 |  | 2.77e-05 (1e-04) | 0 | pass |
| 4 | 16 | 1 | 76 | 121.5 | 503.89 | 557.08 | 0.501 | prepare 0.22, inference 502.99, decode 0.23 | 1586 | 21070 |  | 2.77e-05 (1e-04) | 0 | pass |
| 8 | 1 | 1 | 28 | 110.8 | 97.05 | 110.02 | 0.399 | prepare 0.35, inference 95.94, decode 0.26 | 1585 | 20950 |  | 1.99e-05 (1e-04) | 0 | pass |
| 8 | 4 | 1 | 31 | 110.1 | 258.16 | 396.44 | 0.407 | prepare 0.43, inference 257.86, decode 0.25 | 1589 | 20972 |  | 1.99e-05 (1e-04) | 0 | pass |
| 8 | 16 | 1 | 43 | 108.4 | 1176.86 | 1257.94 | 0.400 | prepare 0.33, inference 1175.38, decode 0.30 | 1589 | 21070 |  | 1.99e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 11 | 84.4 | 188.56 | n/a | 0.297 | prepare 0.59, inference 187.57, decode 0.45 | 1589 | 20950 |  | 2.95e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 14 | 83.9 | 759.32 | n/a | 0.297 | prepare 0.74, inference 750.78, decode 0.52 | 1586 | 20972 |  | 2.95e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 26 | 83.8 | 2678.86 | 3088.09 | 0.301 | prepare 1.05, inference 2571.83, decode 0.49 | 1587 | 21038 |  | 2.95e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 82.2 | 384.77 | n/a | 0.300 | prepare 1.07, inference 379.64, decode 0.87 | 1591 | 20950 |  | 2.51e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 82.3 | 1540.38 | n/a | 0.300 | prepare 4.12, inference 1535.38, decode 0.92 | 1584 | 20972 |  | 2.51e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 21 | 82.7 | 4271.98 | 6196.21 | 0.294 | prepare 5.55, inference 4259.56, decode 0.93 | 1597 | 20998 |  | 2.51e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 81.3 | 810.56 | n/a | 0.292 | prepare 2.16, inference 779.81, decode 2.02 | 1591 | 20948 |  | 2.52e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 81.7 | 3120.87 | n/a | 0.292 | prepare 2.78, inference 3114.96, decode 2.10 | 1589 | 20972 |  | 2.52e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 18 | 81.7 | 7777.98 | n/a | 0.292 | prepare 12.25, inference 7385.84, decode 2.17 | 1596 | 20972 |  | 2.52e-05 (1e-04) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 227.1 | 129.0 | 129.6 |
| 4 | 500.4 | 471.7 | 458.0 |
| 8 | 549.2 | 542.6 | 531.9 |
| 16 | 593.1 | 583.7 | 585.2 |
| 32 | 595.1 | 597.9 | 591.9 |
| 64 | 598.1 | 602.6 | 599.5 |

Best: batch 64 at concurrency 4, 602.6 items/s; best single-item callers: concurrency 1, 227.1 items/s (ratio 2.65).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 455 | 227.1 | 4.18 | 5.38 | 1.000 | prepare 0.01, inference 4.16, decode 0.01 | 1143 | 1442 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 4 | 1 | 262 | 129.0 | 32.32 | 34.85 | 1.000 | prepare 0.02, inference 32.24, decode 0.03 | 1150 | 1484 |  | 0.00e+00 (1e-04) | 0 | pass |
| 1 | 16 | 1 | 275 | 129.6 | 130.70 | 136.90 | 1.000 | prepare 0.02, inference 130.65, decode 0.02 | 1157 | 1614 |  | 0.00e+00 (1e-04) | 0 | pass |
| 4 | 1 | 1 | 251 | 500.4 | 7.97 | 9.32 | 0.746 | prepare 0.03, inference 7.91, decode 0.03 | 1157 | 1492 |  | 1.23e-05 (1e-04) | 0 | pass |
| 4 | 4 | 1 | 240 | 471.7 | 33.60 | 36.56 | 0.746 | prepare 0.05, inference 33.45, decode 0.07 | 1161 | 1516 |  | 1.23e-05 (1e-04) | 0 | pass |
| 4 | 16 | 1 | 245 | 458.0 | 139.51 | 148.13 | 0.746 | prepare 0.05, inference 139.20, decode 0.07 | 1161 | 1614 |  | 1.23e-05 (1e-04) | 0 | pass |
| 8 | 1 | 1 | 138 | 549.2 | 14.31 | 16.30 | 0.659 | prepare 0.06, inference 14.17, decode 0.07 | 1161 | 1492 |  | 1.70e-05 (1e-04) | 0 | pass |
| 8 | 4 | 1 | 139 | 542.6 | 58.64 | 62.54 | 0.660 | prepare 0.10, inference 58.34, decode 0.13 | 1161 | 1516 |  | 1.70e-05 (1e-04) | 0 | pass |
| 8 | 16 | 1 | 149 | 531.9 | 236.83 | 247.57 | 0.660 | prepare 0.11, inference 236.60, decode 0.14 | 1161 | 1614 |  | 1.70e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 75 | 593.1 | 26.51 | 30.93 | 0.625 | prepare 0.12, inference 26.31, decode 0.12 | 1161 | 2508 |  | 2.07e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 77 | 583.7 | 107.41 | 121.07 | 0.626 | prepare 0.21, inference 106.94, decode 0.26 | 1161 | 2540 |  | 2.07e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 89 | 585.2 | 429.83 | 453.65 | 0.626 | prepare 0.21, inference 429.39, decode 0.25 | 1163 | 2638 |  | 2.07e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 38 | 595.1 | 54.81 | 59.39 | 0.596 | prepare 0.21, inference 53.63, decode 0.24 | 1163 | 2518 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 41 | 597.9 | 212.77 | 217.44 | 0.597 | prepare 0.27, inference 212.06, decode 0.25 | 1163 | 2540 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 52 | 591.9 | 854.29 | 879.20 | 0.595 | prepare 0.26, inference 853.71, decode 0.26 | 1165 | 2638 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 19 | 598.1 | 112.04 | n/a | 0.579 | prepare 0.45, inference 111.09, decode 0.47 | 1165 | 3532 |  | 1.80e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 22 | 602.6 | 422.31 | 428.09 | 0.579 | prepare 0.52, inference 420.80, decode 0.53 | 1169 | 3564 |  | 1.80e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 34 | 599.5 | 1693.40 | 1720.46 | 0.579 | prepare 0.55, inference 1691.41, decode 0.50 | 1170 | 3662 |  | 1.80e-05 (1e-04) | 0 | pass |


## gliner2-base-v1 · cuda · baseline · intra-op 4 · max_batch_size default · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 71.1 |
| 16 | 71.3 |
| 32 | 71.1 |
| 64 | 71.0 |

Best: batch 16 at concurrency 1, 71.3 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 18 | 71.1 | 112.37 | n/a | 0.968 | prepare 0.81, inference 110.53, decode 0.86 | 1140 | 3618 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 10 | 71.3 | 225.54 | n/a | 0.965 | prepare 1.78, inference 219.62, decode 1.75 | 1141 | 3618 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 71.1 | 451.14 | n/a | 0.967 | prepare 4.14, inference 438.55, decode 3.50 | 1145 | 3618 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 71.0 | 905.05 | n/a | 0.967 | prepare 6.88, inference 877.69, decode 7.14 | 1151 | 3618 |  | 1.79e-05 (1e-04) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 111.2 |
| 16 | 110.9 |
| 32 | 107.4 |
| 64 | 108.4 |

Best: batch 8 at concurrency 1, 111.2 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 28 | 111.2 | 96.12 | 109.51 | 0.399 | prepare 0.29, inference 95.28, decode 0.23 | 1132 | 3618 |  | 1.99e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 14 | 110.9 | 132.90 | n/a | 0.400 | prepare 0.50, inference 128.09, decode 0.44 | 1134 | 3618 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 107.4 | 327.44 | n/a | 0.409 | prepare 1.08, inference 291.12, decode 0.90 | 1138 | 3618 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 108.4 | 588.49 | n/a | 0.404 | prepare 2.07, inference 583.79, decode 2.05 | 1139 | 3618 |  | 1.99e-05 (1e-04) | 0 | pass |


## gliner2-base-v1 · cuda · micro-w0-r8 · intra-op 4 · max_batch_size call (8) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 59.6 | 68.3 | 72.9 |

Best: batch 1 at concurrency 16, 72.9 items/s; best single-item callers: concurrency 16, 72.9 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 120 | 59.6 | 16.70 | 17.92 | 1.000 | call 16.70 | 1417 | 4400 | 1.00 / 0.03 | 0.00e+00 (2e-05) | 0 | pass |
| 1 | 4 | 1 | 140 | 68.3 | 59.05 | 60.85 | 0.982 | call 59.03 | 1417 | 4400 | 2.00 / 21.84 | 3.81e-06 (2e-05) | 0 | pass |
| 1 | 16 | 1 | 161 | 72.9 | 219.89 | 224.67 | 0.966 | call 219.88 | 1417 | 4400 | 7.67 / 103.48 | 3.93e-06 (2e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 138.4 | 138.9 | 98.5 |

Best: batch 1 at concurrency 4, 138.9 items/s; best single-item callers: concurrency 4, 138.9 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 278 | 138.4 | 5.33 | 16.54 | 1.000 | call 5.33 | 1351 | 2352 | 1.00 / 0.05 | 0.00e+00 (2e-05) | 0 | pass |
| 1 | 4 | 1 | 281 | 138.9 | 19.75 | 55.82 | 0.630 | call 19.75 | 1352 | 2352 | 1.99 / 10.40 | 5.25e-06 (2e-05) | 0 | pass |
| 1 | 16 | 1 | 209 | 98.5 | 176.95 | 216.73 | 0.364 | call 176.95 | 1417 | 4400 | 7.74 / 77.65 | 3.58e-06 (2e-05) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 225.3 | 373.4 | 560.5 |

Best: batch 1 at concurrency 16, 560.5 items/s; best single-item callers: concurrency 16, 560.5 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 451 | 225.3 | 4.23 | 5.56 | 1.000 | call 4.23 | 1340 | 2352 | 1.00 / 0.02 | 0.00e+00 (2e-05) | 0 | pass |
| 1 | 4 | 1 | 749 | 373.4 | 10.50 | 12.09 | 0.828 | call 10.50 | 1347 | 2352 | 2.00 / 4.71 | 1.01e-06 (2e-05) | 0 | pass |
| 1 | 16 | 1 | 1137 | 560.5 | 27.64 | 31.74 | 0.687 | call 27.64 | 1349 | 2352 | 7.95 / 13.93 | 2.32e-06 (2e-05) | 0 | pass |


## gliner2-base-v1 · cuda · micro-w2-r16 · intra-op 4 · max_batch_size call (16) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 54.1 | 68.5 | 74.3 |

Best: batch 1 at concurrency 16, 74.3 items/s; best single-item callers: concurrency 16, 74.3 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 109 | 54.1 | 18.39 | 19.45 | 1.000 | call 18.39 | 1421 | 4396 | 1.00 / 2.00 | 0.00e+00 (2e-05) | 0 | pass |
| 1 | 4 | 1 | 140 | 68.5 | 58.63 | 60.78 | 0.981 | call 58.56 | 1421 | 4396 | 2.00 / 21.77 | 3.81e-06 (2e-05) | 0 | pass |
| 1 | 16 | 1 | 160 | 74.3 | 215.04 | 226.21 | 0.966 | call 215.02 | 1421 | 4396 | 8.00 / 86.93 | 3.46e-06 (2e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 112.5 | 113.9 | 97.6 |

Best: batch 1 at concurrency 4, 113.9 items/s; best single-item callers: concurrency 4, 113.9 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 225 | 112.5 | 6.99 | 18.03 | 1.000 | call 6.99 | 1350 | 2348 | 1.00 / 2.02 | 0.00e+00 (2e-05) | 0 | pass |
| 1 | 4 | 1 | 228 | 113.9 | 45.08 | 58.72 | 0.474 | call 44.92 | 1352 | 2348 | 4.00 / 1.85 | 3.46e-06 (2e-05) | 0 | pass |
| 1 | 16 | 1 | 214 | 97.6 | 183.88 | 216.06 | 0.365 | call 183.69 | 1420 | 4396 | 7.93 / 72.41 | 2.50e-06 (2e-05) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 147.0 | 405.0 | 567.5 |

Best: batch 1 at concurrency 16, 567.5 items/s; best single-item callers: concurrency 16, 567.5 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 295 | 147.0 | 6.48 | 8.82 | 1.000 | call 6.48 | 1341 | 2348 | 1.00 / 2.00 | 0.00e+00 (2e-05) | 0 | pass |
| 1 | 4 | 1 | 812 | 405.0 | 9.79 | 11.20 | 0.751 | call 9.78 | 1348 | 2348 | 4.00 / 1.95 | 2.26e-06 (2e-05) | 0 | pass |
| 1 | 16 | 1 | 1152 | 567.5 | 27.23 | 32.16 | 0.679 | call 27.23 | 1349 | 2348 | 8.00 / 12.93 | 4.05e-06 (2e-05) | 0 | pass |


## gliner2-base-v1 · cuda · sorted · intra-op 4 · max_batch_size 8 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 71.8 | 73.1 | 72.8 |
| 32 | 72.1 | 73.3 | 73.0 |
| 64 | 72.2 | 73.6 | 73.0 |

Best: batch 64 at concurrency 4, 73.6 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 10 | 71.8 | 220.50 | n/a | 0.981 | prepare 1.80, inference 215.57, decode 1.80 | 1174 | 4468 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 13 | 73.1 | 556.03 | n/a | 0.983 | prepare 2.05, inference 549.61, decode 1.77 | 1176 | 4492 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 23 | 72.8 | 2620.04 | 4601.01 | 0.982 | prepare 10.74, inference 2616.89, decode 1.79 | 1169 | 4542 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 72.1 | 437.64 | n/a | 0.987 | prepare 3.58, inference 430.24, decode 3.56 | 1154 | 4470 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 73.3 | 1651.34 | n/a | 0.987 | prepare 6.82, inference 1520.78, decode 3.53 | 1165 | 4492 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 20 | 73.0 | 5130.90 | 8643.27 | 0.987 | prepare 37.97, inference 4763.19, decode 3.61 | 1173 | 4526 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 72.2 | 878.34 | n/a | 0.992 | prepare 7.26, inference 861.90, decode 7.46 | 1166 | 4470 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 73.6 | 3025.08 | n/a | 0.992 | prepare 9.05, inference 2619.40, decode 7.58 | 1186 | 4492 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 16 | 73.0 | 8760.49 | n/a | 0.992 | prepare 152.08, inference 8543.42, decode 7.21 | 1200 | 4542 |  | 1.79e-05 (1e-04) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 138.0 | 137.9 | 137.4 |
| 32 | 201.1 | 202.9 | 201.5 |
| 64 | 211.3 | 214.3 | 213.0 |

Best: batch 64 at concurrency 4, 214.3 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 18 | 138.0 | 113.27 | n/a | 0.515 | prepare 0.52, inference 111.66, decode 0.43 | 1132 | 4418 |  | 2.07e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 21 | 137.9 | 353.91 | 889.73 | 0.514 | prepare 0.80, inference 352.80, decode 0.48 | 1141 | 4460 |  | 2.07e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 32 | 137.4 | 1737.86 | 1918.87 | 0.524 | prepare 0.66, inference 1735.49, decode 0.49 | 1152 | 4590 |  | 2.07e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 13 | 201.1 | 158.36 | n/a | 0.760 | prepare 1.02, inference 156.61, decode 0.88 | 1152 | 4468 |  | 1.76e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 16 | 202.9 | 389.30 | n/a | 0.763 | prepare 1.15, inference 370.92, decode 0.89 | 1153 | 4492 |  | 1.76e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 27 | 201.5 | 1378.28 | 4184.32 | 0.764 | prepare 1.54, inference 1376.45, decode 0.91 | 1156 | 4526 |  | 1.76e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 211.3 | 312.58 | n/a | 0.819 | prepare 2.17, inference 295.28, decode 1.94 | 1149 | 4468 |  | 1.38e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 214.3 | 1023.27 | n/a | 0.819 | prepare 3.02, inference 981.67, decode 1.92 | 1153 | 4492 |  | 1.38e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 21 | 213.0 | 3385.34 | 6066.51 | 0.818 | prepare 23.30, inference 3364.28, decode 2.10 | 1168 | 4550 |  | 1.38e-05 (1e-04) | 0 | pass |


## gliner2-base-v1 · cuda · sorted+tokens4096 · intra-op 4 · max_batch_size call (64) · max_batch_tokens 4096 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 71.2 | 72.4 | 71.7 |
| 16 | 72.3 | 73.2 | 72.7 |
| 32 | 72.0 | 73.3 | 73.0 |
| 64 | 72.2 | 73.4 | 72.8 |

Best: batch 64 at concurrency 4, 73.4 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 18 | 71.2 | 112.66 | n/a | 0.968 | prepare 0.79, inference 110.83, decode 0.87 | 1124 | 4440 |  | 1.79e-05 (1e-04) | 0 | pass |
| 8 | 4 | 1 | 22 | 72.4 | 441.13 | 445.66 | 0.967 | prepare 0.89, inference 438.50, decode 0.87 | 1124 | 4464 |  | 1.79e-05 (1e-04) | 0 | pass |
| 8 | 16 | 1 | 33 | 71.7 | 1767.48 | 1793.29 | 0.967 | prepare 1.44, inference 1765.99, decode 0.93 | 1136 | 4562 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 10 | 72.3 | 219.97 | n/a | 0.981 | prepare 1.73, inference 215.10, decode 1.72 | 1117 | 4440 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 13 | 73.2 | 868.49 | n/a | 0.983 | prepare 2.05, inference 859.26, decode 1.77 | 1115 | 4464 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 23 | 72.7 | 2956.10 | 4070.70 | 0.982 | prepare 11.28, inference 2952.48, decode 1.79 | 1124 | 4528 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 72.0 | 441.91 | n/a | 0.988 | prepare 4.04, inference 432.60, decode 3.68 | 1115 | 4440 |  | 1.88e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 73.3 | 1734.96 | n/a | 0.988 | prepare 7.39, inference 1575.10, decode 4.04 | 1120 | 4464 |  | 1.88e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 20 | 73.0 | 5722.42 | 7798.58 | 0.988 | prepare 35.33, inference 5533.24, decode 3.59 | 1137 | 4512 |  | 1.88e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 72.2 | 881.32 | n/a | 0.992 | prepare 6.99, inference 862.29, decode 7.16 | 1114 | 4440 |  | 1.88e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 73.4 | 2745.31 | n/a | 0.992 | prepare 12.95, inference 2611.71, decode 7.31 | 1141 | 4464 |  | 1.88e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 17 | 72.8 | 10432.96 | n/a | 0.992 | prepare 141.08, inference 10276.30, decode 7.24 | 1181 | 4504 |  | 1.88e-05 (1e-04) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 111.0 | 110.0 | 106.6 |
| 16 | 186.7 | 190.3 | 187.1 |
| 32 | 188.9 | 188.9 | 189.3 |
| 64 | 219.3 | 222.0 | 222.0 |

Best: batch 64 at concurrency 16, 222.0 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 28 | 111.0 | 95.90 | 109.65 | 0.399 | prepare 0.26, inference 95.13, decode 0.23 | 1079 | 4390 |  | 1.99e-05 (1e-04) | 0 | pass |
| 8 | 4 | 1 | 31 | 110.0 | 258.14 | 394.78 | 0.407 | prepare 0.38, inference 257.68, decode 0.31 | 1091 | 4432 |  | 1.99e-05 (1e-04) | 0 | pass |
| 8 | 16 | 1 | 42 | 106.6 | 1175.35 | 1206.57 | 0.399 | prepare 0.35, inference 1174.56, decode 0.27 | 1096 | 4562 |  | 1.99e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 24 | 186.7 | 92.37 | 104.67 | 0.654 | prepare 0.50, inference 87.17, decode 0.46 | 1096 | 4440 |  | 1.73e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 27 | 190.3 | 310.79 | 597.06 | 0.653 | prepare 0.72, inference 309.71, decode 0.56 | 1100 | 4464 |  | 1.73e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 38 | 187.1 | 772.85 | 2088.61 | 0.647 | prepare 0.63, inference 768.68, decode 0.53 | 1105 | 4562 |  | 1.73e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 12 | 188.9 | 170.85 | n/a | 0.655 | prepare 1.06, inference 161.52, decode 0.87 | 1109 | 4440 |  | 2.01e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 15 | 188.9 | 457.40 | n/a | 0.655 | prepare 1.19, inference 454.99, decode 0.89 | 1103 | 4464 |  | 2.01e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 24 | 189.3 | 2238.30 | 3293.21 | 0.655 | prepare 4.41, inference 2142.33, decode 1.03 | 1112 | 4520 |  | 2.01e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 219.3 | 296.51 | n/a | 0.777 | prepare 2.14, inference 285.57, decode 2.03 | 1113 | 4440 |  | 1.53e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 222.0 | 1162.42 | n/a | 0.777 | prepare 3.63, inference 1032.06, decode 2.07 | 1108 | 4464 |  | 1.53e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 21 | 222.0 | 3760.47 | 5282.46 | 0.775 | prepare 13.92, inference 3745.82, decode 2.08 | 1124 | 4520 |  | 1.53e-05 (1e-04) | 0 | pass |


## gliner2-base-v1 · cuda · tokens4096 · intra-op 4 · max_batch_size call (64) · max_batch_tokens 4096 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 70.7 | 72.3 | 71.7 |
| 16 | 71.1 | 72.0 | 71.7 |
| 32 | 71.0 | 72.0 | 72.0 |
| 64 | 70.9 | 72.1 | 71.7 |

Best: batch 8 at concurrency 4, 72.3 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 18 | 70.7 | 113.00 | n/a | 0.968 | prepare 0.93, inference 110.96, decode 1.00 | 1171 | 4564 |  | 1.79e-05 (1e-04) | 0 | pass |
| 8 | 4 | 1 | 21 | 72.3 | 440.00 | 447.47 | 0.968 | prepare 1.03, inference 437.88, decode 1.01 | 1175 | 4596 |  | 1.79e-05 (1e-04) | 0 | pass |
| 8 | 16 | 1 | 33 | 71.7 | 1766.76 | 1789.98 | 0.967 | prepare 1.10, inference 1765.30, decode 1.00 | 1174 | 4694 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 10 | 71.1 | 225.62 | n/a | 0.965 | prepare 1.87, inference 219.60, decode 1.94 | 1166 | 4572 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 13 | 72.0 | 857.12 | n/a | 0.967 | prepare 2.52, inference 850.37, decode 1.98 | 1165 | 4596 |  | 1.79e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 23 | 71.7 | 2229.77 | 4893.33 | 0.967 | prepare 7.56, inference 2219.90, decode 2.00 | 1176 | 4654 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 71.0 | 452.07 | n/a | 0.967 | prepare 3.68, inference 438.44, decode 3.89 | 1180 | 4574 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 72.0 | 1756.87 | n/a | 0.967 | prepare 6.46, inference 1700.51, decode 3.98 | 1173 | 4596 |  | 1.79e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 19 | 72.0 | 4779.14 | n/a | 0.967 | prepare 37.27, inference 4734.69, decode 4.05 | 1188 | 4638 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 70.9 | 905.49 | n/a | 0.967 | prepare 7.29, inference 877.53, decode 7.93 | 1168 | 4572 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 72.1 | 3878.71 | n/a | 0.967 | prepare 9.91, inference 3513.50, decode 7.97 | 1185 | 4596 |  | 1.79e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 16 | 71.7 | 11177.52 | n/a | 0.967 | prepare 140.45, inference 10974.31, decode 8.15 | 1225 | 4694 |  | 1.79e-05 (1e-04) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 110.6 | 109.7 | 106.5 |
| 16 | 102.3 | 100.9 | 101.3 |
| 32 | 106.3 | 106.4 | 106.1 |
| 64 | 106.1 | 106.6 | 106.5 |

Best: batch 8 at concurrency 1, 110.6 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 28 | 110.6 | 96.84 | 109.56 | 0.399 | prepare 0.30, inference 95.36, decode 0.26 | 1132 | 3498 |  | 1.99e-05 (1e-04) | 0 | pass |
| 8 | 4 | 1 | 31 | 109.7 | 259.70 | 395.43 | 0.407 | prepare 0.32, inference 259.39, decode 0.28 | 1141 | 3540 |  | 1.99e-05 (1e-04) | 0 | pass |
| 8 | 16 | 1 | 42 | 106.5 | 1178.37 | 1204.53 | 0.399 | prepare 0.34, inference 1177.45, decode 0.29 | 1145 | 3670 |  | 1.99e-05 (1e-04) | 0 | pass |
| 16 | 1 | 1 | 13 | 102.3 | 127.47 | n/a | 0.367 | prepare 0.59, inference 126.55, decode 0.50 | 1145 | 3550 |  | 1.69e-05 (1e-04) | 0 | pass |
| 16 | 4 | 1 | 16 | 100.9 | 635.19 | n/a | 0.375 | prepare 0.62, inference 627.14, decode 0.54 | 1145 | 3572 |  | 1.69e-05 (1e-04) | 0 | pass |
| 16 | 16 | 1 | 27 | 101.3 | 2257.98 | 3246.51 | 0.372 | prepare 0.77, inference 2256.70, decode 0.54 | 1151 | 3646 |  | 1.69e-05 (1e-04) | 0 | pass |
| 32 | 1 | 1 | 10 | 106.3 | 316.64 | n/a | 0.399 | prepare 1.05, inference 308.98, decode 0.96 | 1154 | 3550 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 4 | 1 | 10 | 106.4 | 792.58 | n/a | 0.399 | prepare 1.35, inference 681.82, decode 1.00 | 1154 | 3572 |  | 1.99e-05 (1e-04) | 0 | pass |
| 32 | 16 | 1 | 21 | 106.1 | 3198.27 | 6161.44 | 0.389 | prepare 1.80, inference 3192.98, decode 1.01 | 1164 | 3606 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 1 | 1 | 10 | 106.1 | 609.05 | n/a | 0.388 | prepare 2.16, inference 597.10, decode 2.23 | 1167 | 3550 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 4 | 1 | 10 | 106.6 | 2393.38 | n/a | 0.388 | prepare 5.99, inference 2087.21, decode 2.51 | 1169 | 3572 |  | 1.99e-05 (1e-04) | 0 | pass |
| 64 | 16 | 1 | 17 | 106.5 | 5765.87 | n/a | 0.387 | prepare 18.93, inference 5745.70, decode 2.34 | 1169 | 3614 |  | 1.99e-05 (1e-04) | 0 | pass |


## multilingual-e5-small · cuda · baseline · intra-op 4 · max_batch_size 8 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 254.5 | 337.5 | 320.0 |
| 32 | 254.6 | 329.4 | 304.5 |
| 64 | 256.8 | 330.7 | 280.2 |

Best: batch 16 at concurrency 4, 337.5 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 32 | 254.5 | 62.33 | 66.63 | 0.905 | tokenize 9.61, encode 52.99 | 1154 | 6774 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 44 | 337.5 | 183.35 | 207.25 | 0.906 | tokenize 14.39, encode 170.03 | 1160 | 6798 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 48 | 320.0 | 762.30 | 951.36 | 0.905 | tokenize 34.56, encode 712.59 | 1122 | 6894 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 16 | 254.6 | 125.92 | n/a | 0.905 | tokenize 18.80, encode 106.77 | 1118 | 6774 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 22 | 329.4 | 384.90 | 428.32 | 0.904 | tokenize 39.92, encode 346.04 | 1097 | 6798 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 30 | 304.5 | 1561.57 | 2134.75 | 0.904 | tokenize 253.50, encode 1410.08 | 1110 | 7918 |  | 7.24e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 256.8 | 247.86 | n/a | 0.905 | tokenize 34.25, encode 213.70 | 1111 | 7806 |  | 7.24e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 12 | 330.7 | 768.31 | n/a | 0.905 | tokenize 64.46, encode 701.15 | 1135 | 7822 |  | 7.24e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 16 | 280.2 | 3614.33 | n/a | 0.905 | tokenize 664.59, encode 2940.83 | 1105 | 7918 |  | 7.24e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 386.8 | 481.8 | 447.9 |
| 32 | 387.0 | 477.2 | 465.7 |
| 64 | 387.5 | 476.2 | 441.2 |

Best: batch 16 at concurrency 4, 481.8 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 49 | 386.8 | 40.21 | 52.85 | 0.419 | tokenize 3.41, encode 36.32 | 790 | 1442 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 62 | 481.8 | 132.58 | 154.02 | 0.419 | tokenize 5.79, encode 126.13 | 843 | 2542 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 67 | 447.9 | 540.74 | 730.65 | 0.417 | tokenize 14.53, encode 518.25 | 1094 | 6894 |  | 9.31e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 25 | 387.0 | 81.21 | 88.86 | 0.422 | tokenize 7.46, encode 73.36 | 1094 | 6774 |  | 9.31e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 32 | 477.2 | 263.38 | 285.63 | 0.420 | tokenize 9.35, encode 251.51 | 1135 | 6798 |  | 9.31e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 34 | 465.7 | 1053.28 | 1204.25 | 0.421 | tokenize 21.32, encode 1037.40 | 1142 | 6894 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 13 | 387.5 | 165.72 | n/a | 0.419 | tokenize 12.92, encode 151.53 | 1146 | 6774 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 16 | 476.2 | 528.65 | n/a | 0.420 | tokenize 28.82, encode 501.69 | 1156 | 6798 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 17 | 441.2 | 2222.89 | n/a | 0.419 | tokenize 177.68, encode 2056.97 | 1153 | 6894 |  | 9.31e-08 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · baseline · intra-op 4 · max_batch_size call (1) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 190.1 | 308.4 | 350.9 |

Best: batch 1 at concurrency 16, 350.9 items/s; best single-item callers: concurrency 16, 350.9 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 381 | 190.1 | 5.30 | 5.94 | 1.000 | tokenize 0.59, encode 4.72 | 797 | 1654 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 619 | 308.4 | 12.51 | 15.81 | 1.000 | tokenize 0.89, encode 11.65 | 793 | 1678 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 715 | 350.9 | 44.61 | 61.46 | 1.000 | tokenize 1.28, encode 43.06 | 789 | 1774 |  | 0.00e+00 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 331.8 | 723.6 | 840.4 |

Best: batch 1 at concurrency 16, 840.4 items/s; best single-item callers: concurrency 16, 840.4 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 664 | 331.8 | 2.52 | 5.42 | 1.000 | tokenize 0.15, encode 2.37 | 776 | 1158 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 1449 | 723.6 | 5.08 | 9.15 | 1.000 | tokenize 0.16, encode 4.86 | 780 | 1166 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 1701 | 840.4 | 18.14 | 26.83 | 1.000 | tokenize 0.31, encode 17.57 | 797 | 1774 |  | 0.00e+00 (1e-05) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 444.2 | 1246.6 | 1782.4 |

Best: batch 1 at concurrency 16, 1782.4 items/s; best single-item callers: concurrency 16, 1782.4 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 889 | 444.2 | 2.20 | 2.49 | 1.000 | tokenize 0.03, encode 2.16 | 756 | 930 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 2507 | 1246.6 | 3.15 | 3.65 | 1.000 | tokenize 0.03, encode 3.11 | 762 | 1006 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 3585 | 1782.4 | 8.04 | 12.96 | 1.000 | tokenize 0.04, encode 7.99 | 776 | 1262 |  | 0.00e+00 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · baseline · intra-op 4 · max_batch_size call (64) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 189.9 | 316.1 | 348.7 |
| 4 | 244.3 | 333.2 | 326.6 |
| 8 | 249.9 | 339.1 | 319.9 |
| 16 | 258.8 | 338.5 | 300.3 |
| 32 | 263.1 | 328.4 | 279.2 |
| 64 | 272.1 | 304.7 | 240.8 |

Best: batch 1 at concurrency 16, 348.7 items/s; best single-item callers: concurrency 16, 348.7 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 380 | 189.9 | 5.28 | 5.92 | 1.000 | tokenize 0.62, encode 4.70 | 839 | 1764 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 634 | 316.1 | 12.54 | 15.26 | 1.000 | tokenize 0.85, encode 11.66 | 840 | 1676 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 712 | 348.7 | 44.62 | 67.50 | 1.000 | tokenize 1.40, encode 42.89 | 846 | 1772 |  | 0.00e+00 (1e-05) | 0 | pass |
| 4 | 1 | 1 | 123 | 244.3 | 16.39 | 17.65 | 0.922 | tokenize 2.57, encode 13.77 | 846 | 1764 |  | 7.45e-08 (1e-05) | 0 | pass |
| 4 | 4 | 1 | 169 | 333.2 | 47.02 | 53.42 | 0.921 | tokenize 4.26, encode 42.67 | 879 | 2700 |  | 7.45e-08 (1e-05) | 0 | pass |
| 4 | 16 | 1 | 173 | 326.6 | 187.52 | 277.81 | 0.921 | tokenize 8.91, encode 178.96 | 949 | 4844 |  | 7.45e-08 (1e-05) | 0 | pass |
| 8 | 1 | 1 | 63 | 249.9 | 32.02 | 33.64 | 0.906 | tokenize 5.42, encode 26.70 | 949 | 4836 |  | 7.24e-08 (1e-05) | 0 | pass |
| 8 | 4 | 1 | 88 | 339.1 | 94.25 | 103.52 | 0.907 | tokenize 6.75, encode 86.01 | 972 | 4748 |  | 7.24e-08 (1e-05) | 0 | pass |
| 8 | 16 | 1 | 90 | 319.9 | 376.09 | 575.85 | 0.907 | tokenize 19.76, encode 356.91 | 1056 | 6892 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 33 | 258.8 | 61.58 | 63.90 | 0.893 | tokenize 10.06, encode 51.45 | 1068 | 6884 |  | 8.94e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 45 | 338.5 | 184.85 | 205.23 | 0.894 | tokenize 11.10, encode 172.32 | 1117 | 6796 |  | 8.94e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 50 | 300.3 | 880.59 | 1081.48 | 0.894 | tokenize 127.73, encode 718.80 | 1278 | 9964 |  | 8.94e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 17 | 263.1 | 121.55 | n/a | 0.887 | tokenize 19.85, encode 101.72 | 1302 | 9884 |  | 8.20e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 23 | 328.4 | 370.18 | 427.61 | 0.887 | tokenize 23.32, encode 347.90 | 1401 | 9868 |  | 8.20e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 30 | 279.2 | 1850.77 | 2626.02 | 0.887 | tokenize 289.00, encode 1527.70 | 1706 | 10988 |  | 8.20e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 272.1 | 235.10 | n/a | 0.886 | tokenize 35.03, encode 199.71 | 1627 | 12908 |  | 8.94e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 11 | 304.7 | 748.77 | n/a | 0.887 | tokenize 61.11, encode 701.12 | 2012 | 19084 |  | 8.94e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 16 | 240.8 | 4024.81 | n/a | 0.886 | tokenize 662.19, encode 3371.50 | 2369 | 19180 |  | 8.94e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 330.9 | 730.8 | 857.6 |
| 4 | 417.1 | 572.2 | 572.2 |
| 8 | 382.0 | 482.2 | 470.2 |
| 16 | 321.6 | 373.8 | 357.0 |
| 32 | 308.5 | 351.1 | 337.2 |
| 64 | 318.0 | 343.2 | 308.4 |

Best: batch 1 at concurrency 16, 857.6 items/s; best single-item callers: concurrency 16, 857.6 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 662 | 330.9 | 2.51 | 5.40 | 1.000 | tokenize 0.15, encode 2.34 | 2208 | 19060 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 1465 | 730.8 | 5.10 | 9.15 | 1.000 | tokenize 0.16, encode 4.87 | 2208 | 19084 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 1742 | 857.6 | 17.97 | 26.60 | 1.000 | tokenize 0.29, encode 17.48 | 2230 | 19180 |  | 0.00e+00 (1e-05) | 0 | pass |
| 4 | 1 | 1 | 209 | 417.1 | 9.80 | 15.36 | 0.508 | tokenize 0.82, encode 8.77 | 2231 | 19068 |  | 9.13e-08 (1e-05) | 0 | pass |
| 4 | 4 | 1 | 287 | 572.2 | 28.66 | 37.51 | 0.508 | tokenize 1.52, encode 26.97 | 2254 | 19084 |  | 9.13e-08 (1e-05) | 0 | pass |
| 4 | 16 | 1 | 300 | 572.2 | 108.97 | 141.62 | 0.510 | tokenize 1.75, encode 106.85 | 2336 | 19180 |  | 9.13e-08 (1e-05) | 0 | pass |
| 8 | 1 | 1 | 96 | 382.0 | 23.81 | 30.07 | 0.420 | tokenize 1.81, encode 21.98 | 2336 | 19060 |  | 9.31e-08 (1e-05) | 0 | pass |
| 8 | 4 | 1 | 122 | 482.2 | 68.55 | 80.70 | 0.419 | tokenize 2.43, encode 64.74 | 2375 | 19084 |  | 9.31e-08 (1e-05) | 0 | pass |
| 8 | 16 | 1 | 127 | 470.2 | 265.16 | 407.53 | 0.421 | tokenize 5.00, encode 262.00 | 2512 | 19180 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 41 | 321.6 | 49.58 | 55.62 | 0.337 | tokenize 3.50, encode 47.12 | 2512 | 19156 |  | 8.94e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 49 | 373.8 | 167.02 | 180.06 | 0.338 | tokenize 4.83, encode 162.37 | 2560 | 19084 |  | 8.94e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 55 | 357.0 | 703.97 | 1034.49 | 0.338 | tokenize 12.04, encode 682.54 | 2753 | 19180 |  | 8.94e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 20 | 308.5 | 105.13 | 107.15 | 0.321 | tokenize 7.67, encode 98.73 | 2753 | 19172 |  | 1.04e-07 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 24 | 351.1 | 352.69 | 571.42 | 0.321 | tokenize 11.74, encode 342.50 | 2801 | 19084 |  | 1.04e-07 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 32 | 337.2 | 1472.64 | 1866.40 | 0.321 | tokenize 66.82, encode 1415.95 | 2705 | 19180 |  | 1.04e-07 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 318.0 | 205.91 | n/a | 0.320 | tokenize 13.02, encode 186.17 | 2423 | 19172 |  | 1.07e-07 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 14 | 343.2 | 718.07 | n/a | 0.320 | tokenize 22.03, encode 689.89 | 2368 | 19084 |  | 1.07e-07 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 16 | 308.4 | 3127.97 | n/a | 0.320 | tokenize 154.27, encode 2968.51 | 2370 | 19180 |  | 1.07e-07 (1e-05) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 465.1 | 1318.9 | 1937.5 |
| 4 | 1578.9 | 3406.6 | 1781.5 |
| 8 | 3158.8 | 6634.1 | 4233.2 |
| 16 | 4474.5 | 7863.5 | 7995.2 |
| 32 | 5226.7 | 8259.3 | 8498.2 |
| 64 | 5960.1 | 8507.4 | 8446.7 |

Best: batch 64 at concurrency 4, 8507.4 items/s; best single-item callers: concurrency 16, 1937.5 items/s (ratio 4.39).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 931 | 465.1 | 2.04 | 2.32 | 1.000 | tokenize 0.03, encode 2.01 | 790 | 1440 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 2656 | 1318.9 | 3.03 | 3.54 | 1.000 | tokenize 0.03, encode 3.00 | 794 | 1516 |  | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 3901 | 1937.5 | 7.36 | 11.74 | 1.000 | tokenize 0.04, encode 7.31 | 810 | 1772 |  | 0.00e+00 (1e-05) | 0 | pass |
| 4 | 1 | 1 | 790 | 1578.9 | 2.49 | 2.68 | 0.776 | tokenize 0.09, encode 2.40 | 810 | 1652 |  | 8.38e-08 (1e-05) | 0 | pass |
| 4 | 4 | 1 | 1709 | 3406.6 | 4.87 | 5.81 | 0.777 | tokenize 0.10, encode 4.77 | 814 | 1676 |  | 8.38e-08 (1e-05) | 0 | pass |
| 4 | 16 | 1 | 904 | 1781.5 | 38.59 | 48.14 | 0.777 | tokenize 0.22, encode 38.40 | 815 | 1772 |  | 8.38e-08 (1e-05) | 0 | pass |
| 8 | 1 | 1 | 790 | 3158.8 | 2.45 | 2.79 | 0.702 | tokenize 0.17, encode 2.27 | 815 | 1684 |  | 8.20e-08 (1e-05) | 0 | pass |
| 8 | 4 | 1 | 1662 | 6634.1 | 4.73 | 5.48 | 0.702 | tokenize 0.18, encode 4.53 | 812 | 1676 |  | 8.20e-08 (1e-05) | 0 | pass |
| 8 | 16 | 1 | 1075 | 4233.2 | 37.97 | 51.30 | 0.702 | tokenize 0.40, encode 37.62 | 813 | 1772 |  | 8.20e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 560 | 4474.5 | 3.56 | 4.03 | 0.642 | tokenize 0.34, encode 3.18 | 813 | 1652 |  | 1.04e-07 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 986 | 7863.5 | 8.01 | 9.50 | 0.642 | tokenize 0.43, encode 7.53 | 817 | 1676 |  | 1.04e-07 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 1013 | 7995.2 | 33.05 | 47.32 | 0.642 | tokenize 0.78, encode 32.19 | 801 | 1772 |  | 1.04e-07 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 327 | 5226.7 | 6.22 | 6.85 | 0.587 | tokenize 0.67, encode 5.55 | 805 | 1676 |  | 9.69e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 518 | 8259.3 | 15.29 | 17.77 | 0.588 | tokenize 1.20, encode 14.15 | 805 | 1676 |  | 9.69e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 543 | 8498.2 | 58.48 | 84.10 | 0.588 | tokenize 1.61, encode 56.71 | 824 | 1772 |  | 9.69e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 187 | 5960.1 | 10.70 | 11.43 | 0.548 | tokenize 1.40, encode 9.26 | 817 | 1756 |  | 8.75e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 269 | 8507.4 | 29.72 | 34.13 | 0.548 | tokenize 2.56, encode 27.22 | 800 | 1676 |  | 8.75e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 266 | 8446.7 | 118.05 | 147.42 | 0.548 | tokenize 4.38, encode 114.08 | 839 | 1772 |  | 8.75e-08 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · baseline · intra-op 4 · max_batch_size default · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 257.0 |
| 16 | 262.1 |
| 32 | 269.0 |
| 64 | 268.3 |

Best: batch 32 at concurrency 1, 269.0 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 65 | 257.0 | 30.91 | 32.96 | 0.905 | tokenize 4.60, encode 26.56 | 851 | 2400 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 33 | 262.1 | 60.93 | 62.95 | 0.893 | tokenize 9.36, encode 51.66 | 851 | 2400 |  | 8.94e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 17 | 269.0 | 119.01 | n/a | 0.887 | tokenize 17.62, encode 101.47 | 851 | 2400 |  | 8.20e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 268.3 | 238.07 | n/a | 0.888 | tokenize 34.35, encode 203.94 | 852 | 2400 |  | 8.20e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 |
| ---: | ---: |
| 8 | 387.4 |
| 16 | 322.9 |
| 32 | 313.8 |
| 64 | 313.4 |

Best: batch 8 at concurrency 1, 387.4 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 97 | 387.4 | 23.25 | 29.71 | 0.420 | tokenize 1.69, encode 21.79 | 795 | 1376 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 41 | 322.9 | 49.37 | 55.36 | 0.337 | tokenize 3.39, encode 46.86 | 796 | 1376 |  | 8.94e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 20 | 313.8 | 104.13 | 105.73 | 0.321 | tokenize 6.77, encode 96.59 | 850 | 2400 |  | 1.04e-07 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 313.4 | 208.54 | n/a | 0.321 | tokenize 13.03, encode 193.22 | 850 | 2400 |  | 1.04e-07 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · micro-w0-r8 · intra-op 4 · max_batch_size call (8) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 166.0 | 261.5 | 297.6 |

Best: batch 1 at concurrency 16, 297.6 items/s; best single-item callers: concurrency 16, 297.6 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 332 | 166.0 | 5.97 | 7.34 | 1.000 | call 5.97 | 844 | 1354 | 1.00 / 0.02 | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 525 | 261.5 | 15.36 | 15.99 | 0.948 | call 15.36 | 845 | 1354 | 2.00 / 4.83 | 6.71e-08 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 609 | 297.6 | 54.05 | 55.29 | 0.908 | call 54.05 | 845 | 1354 | 7.91 / 23.50 | 6.71e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 289.9 | 427.5 | 383.8 |

Best: batch 1 at concurrency 4, 427.5 items/s; best single-item callers: concurrency 4, 427.5 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 580 | 289.9 | 2.86 | 6.09 | 1.000 | call 2.86 | 830 | 842 | 1.00 / 0.02 | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 858 | 427.5 | 8.96 | 15.90 | 0.634 | call 8.95 | 847 | 1354 | 2.00 / 3.95 | 8.34e-08 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 777 | 383.8 | 44.16 | 52.44 | 0.388 | call 44.16 | 848 | 1354 | 7.93 / 19.83 | 9.69e-08 (1e-05) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 404.1 | 786.3 | 3191.5 |

Best: batch 1 at concurrency 16, 3191.5 items/s; best single-item callers: concurrency 16, 3191.5 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 809 | 404.1 | 2.38 | 3.17 | 1.000 | call 2.38 | 823 | 842 | 1.00 / 0.02 | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 1575 | 786.3 | 5.03 | 5.34 | 0.877 | call 5.03 | 831 | 842 | 2.00 / 2.40 | 8.20e-08 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 5000 | 3191.5 | 4.90 | 5.54 | 0.711 | call 4.90 | 832 | 842 | 7.96 / 2.30 | 8.20e-08 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · micro-w2-r16 · intra-op 4 · max_batch_size call (16) · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 121.2 | 261.1 | 296.8 |

Best: batch 1 at concurrency 16, 296.8 items/s; best single-item callers: concurrency 16, 296.8 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 243 | 121.2 | 8.21 | 9.24 | 1.000 | call 8.21 | 821 | 1452 | 1.00 / 2.00 | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 525 | 261.1 | 15.36 | 16.12 | 0.947 | call 15.36 | 821 | 1452 | 2.02 / 4.77 | 6.71e-08 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 608 | 296.8 | 53.69 | 55.41 | 0.904 | call 53.69 | 823 | 1452 | 8.00 / 19.37 | 7.24e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 166.9 | 409.7 | 399.4 |

Best: batch 1 at concurrency 4, 409.7 items/s; best single-item callers: concurrency 4, 409.7 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 334 | 166.9 | 5.80 | 8.40 | 1.000 | call 5.79 | 806 | 940 | 1.00 / 2.00 | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 821 | 409.7 | 9.03 | 15.29 | 0.581 | call 9.02 | 824 | 1452 | 2.46 / 2.76 | 9.13e-08 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 811 | 399.4 | 40.20 | 52.00 | 0.406 | call 40.20 | 820 | 1452 | 7.95 / 18.05 | 1.04e-07 (1e-05) | 0 | pass |

### short: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 1 | 213.3 | 831.6 | 3902.0 |

Best: batch 1 at concurrency 16, 3902.0 items/s; best single-item callers: concurrency 16, 3902.0 items/s (ratio 1.00).

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | 427 | 213.3 | 4.41 | 6.04 | 1.000 | call 4.41 | 798 | 940 | 1.00 / 2.01 | 0.00e+00 (1e-05) | 0 | pass |
| 1 | 4 | 1 | 1665 | 831.6 | 4.56 | 6.72 | 0.780 | call 4.56 | 801 | 940 | 3.96 / 1.95 | 8.38e-08 (1e-05) | 0 | pass |
| 1 | 16 | 1 | 5000 | 3902.0 | 3.79 | 5.69 | 0.656 | call 3.78 | 806 | 940 | 13.89 / 0.56 | 1.04e-07 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · sorted · intra-op 4 · max_batch_size 8 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 263.8 | 356.7 | 333.0 |
| 32 | 273.2 | 363.6 | 321.4 |
| 64 | 276.1 | 363.0 | 309.6 |

Best: batch 32 at concurrency 4, 363.6 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 33 | 263.8 | 60.54 | 63.49 | 0.947 | tokenize 10.06, encode 50.39 | 1118 | 5748 |  | 8.20e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 47 | 356.7 | 175.12 | 199.58 | 0.947 | tokenize 15.12, encode 163.71 | 1142 | 5772 |  | 8.20e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 49 | 333.0 | 748.97 | 891.07 | 0.947 | tokenize 49.71, encode 679.86 | 1153 | 6892 |  | 8.20e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 18 | 273.2 | 116.07 | n/a | 0.968 | tokenize 18.55, encode 98.06 | 1156 | 6772 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 25 | 363.6 | 350.99 | 384.46 | 0.968 | tokenize 31.44, encode 317.88 | 1095 | 6796 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 33 | 321.4 | 1626.13 | 1669.27 | 0.968 | tokenize 263.01, encode 1362.75 | 1099 | 7916 |  | 7.24e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 276.1 | 231.80 | n/a | 0.985 | tokenize 36.50, encode 193.97 | 1073 | 7908 |  | 8.20e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 13 | 363.0 | 679.94 | n/a | 0.985 | tokenize 68.25, encode 617.66 | 1080 | 7820 |  | 8.20e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 16 | 309.6 | 3264.79 | n/a | 0.985 | tokenize 643.17, encode 2604.35 | 1093 | 7916 |  | 8.20e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 16 | 486.4 | 609.5 | 591.3 |
| 32 | 628.1 | 855.4 | 854.6 |
| 64 | 700.7 | 986.3 | 971.9 |

Best: batch 64 at concurrency 4, 986.3 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 16 | 1 | 1 | 61 | 486.4 | 32.90 | 36.41 | 0.554 | tokenize 3.45, encode 29.77 | 800 | 1440 |  | 7.82e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 80 | 609.5 | 102.64 | 124.35 | 0.554 | tokenize 5.56, encode 96.52 | 835 | 2540 |  | 7.82e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 87 | 591.3 | 400.54 | 528.30 | 0.553 | tokenize 9.68, encode 389.76 | 1026 | 5868 |  | 7.82e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 40 | 628.1 | 52.27 | 54.70 | 0.745 | tokenize 7.03, encode 44.81 | 1022 | 5748 |  | 8.20e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 56 | 855.4 | 146.05 | 163.09 | 0.745 | tokenize 12.34, encode 134.28 | 1058 | 5772 |  | 8.20e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 66 | 854.6 | 575.68 | 651.64 | 0.745 | tokenize 19.48, encode 556.36 | 1097 | 5868 |  | 8.20e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 22 | 700.7 | 91.22 | 92.26 | 0.855 | tokenize 13.64, encode 77.55 | 1097 | 5756 |  | 8.94e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 34 | 986.3 | 253.76 | 287.88 | 0.855 | tokenize 24.83, encode 232.37 | 1150 | 5772 |  | 8.94e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 42 | 971.9 | 951.24 | 1163.02 | 0.855 | tokenize 49.20, encode 901.57 | 1118 | 5868 |  | 8.94e-08 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · sorted+tokens4096 · intra-op 4 · max_batch_size call (64) · max_batch_tokens 4096 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 254.3 | 338.7 | 324.0 |
| 16 | 262.9 | 354.6 | 329.3 |
| 32 | 268.3 | 352.5 | 330.8 |
| 64 | 275.3 | 341.9 | 301.3 |

Best: batch 16 at concurrency 4, 354.6 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 64 | 254.3 | 31.54 | 33.15 | 0.905 | tokenize 4.97, encode 26.58 | 1074 | 6742 |  | 7.24e-08 (1e-05) | 0 | pass |
| 8 | 4 | 1 | 86 | 338.7 | 94.59 | 106.33 | 0.906 | tokenize 7.11, encode 86.29 | 1078 | 6734 |  | 7.24e-08 (1e-05) | 0 | pass |
| 8 | 16 | 1 | 93 | 324.0 | 392.11 | 537.47 | 0.906 | tokenize 22.14, encode 361.47 | 1081 | 7854 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 33 | 262.9 | 60.73 | 62.91 | 0.948 | tokenize 10.12, encode 50.89 | 1081 | 7838 |  | 6.71e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 47 | 354.6 | 177.71 | 195.08 | 0.948 | tokenize 15.72, encode 161.73 | 1097 | 7758 |  | 6.71e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 48 | 329.3 | 713.38 | 883.21 | 0.948 | tokenize 33.80, encode 676.03 | 1090 | 7854 |  | 6.71e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 17 | 268.3 | 118.08 | n/a | 0.970 | tokenize 18.83, encode 99.38 | 1086 | 7734 |  | 6.71e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 24 | 352.5 | 354.52 | 395.02 | 0.970 | tokenize 33.80, encode 321.41 | 1080 | 7758 |  | 6.71e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 32 | 330.8 | 1450.44 | 1730.09 | 0.970 | tokenize 149.80, encode 1321.04 | 1101 | 8878 |  | 6.71e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 275.3 | 230.01 | n/a | 0.985 | tokenize 34.54, encode 195.59 | 1101 | 8758 |  | 6.52e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 12 | 341.9 | 755.05 | n/a | 0.985 | tokenize 99.27, encode 653.38 | 1084 | 8782 |  | 6.52e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 16 | 301.3 | 3293.12 | n/a | 0.985 | tokenize 622.30, encode 2696.18 | 1153 | 8878 |  | 6.52e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 386.6 | 464.7 | 459.6 |
| 16 | 525.9 | 707.1 | 703.6 |
| 32 | 549.4 | 718.8 | 701.5 |
| 64 | 661.5 | 874.2 | 824.3 |

Best: batch 64 at concurrency 4, 874.2 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 97 | 386.6 | 23.30 | 29.53 | 0.420 | tokenize 1.69, encode 21.87 | 760 | 1378 |  | 9.31e-08 (1e-05) | 0 | pass |
| 8 | 4 | 1 | 118 | 464.7 | 69.49 | 97.89 | 0.421 | tokenize 2.55, encode 67.09 | 849 | 3502 |  | 9.31e-08 (1e-05) | 0 | pass |
| 8 | 16 | 1 | 118 | 459.6 | 264.32 | 411.39 | 0.421 | tokenize 4.99, encode 261.62 | 1017 | 5790 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 66 | 525.9 | 30.92 | 37.73 | 0.560 | tokenize 3.54, encode 25.74 | 1021 | 5742 |  | 9.78e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 90 | 707.1 | 89.02 | 105.73 | 0.561 | tokenize 6.19, encode 83.57 | 1061 | 5710 |  | 9.78e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 96 | 703.6 | 357.48 | 435.50 | 0.563 | tokenize 13.49, encode 343.12 | 1074 | 5806 |  | 9.78e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 35 | 549.4 | 57.55 | 62.81 | 0.601 | tokenize 7.28, encode 50.72 | 1075 | 5686 |  | 1.04e-07 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 48 | 718.8 | 175.78 | 191.86 | 0.599 | tokenize 12.45, encode 163.43 | 1087 | 5710 |  | 1.04e-07 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 49 | 701.5 | 695.99 | 817.73 | 0.600 | tokenize 31.15, encode 675.73 | 1075 | 5806 |  | 1.04e-07 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 21 | 661.5 | 95.05 | 99.68 | 0.702 | tokenize 13.73, encode 80.64 | 1076 | 5686 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 30 | 874.2 | 288.39 | 316.45 | 0.702 | tokenize 26.18, encode 262.21 | 1041 | 5710 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 32 | 824.3 | 1144.02 | 1465.52 | 0.702 | tokenize 92.71, encode 1092.17 | 1074 | 6830 |  | 9.31e-08 (1e-05) | 0 | pass |


## multilingual-e5-small · cuda · tokens4096 · intra-op 4 · max_batch_size call (64) · max_batch_tokens 4096 · NVIDIA GeForce RTX 3090

### long: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 255.6 | 340.0 | 331.6 |
| 16 | 257.6 | 338.3 | 322.0 |
| 32 | 258.3 | 335.3 | 312.1 |
| 64 | 257.3 | 326.6 | 277.6 |

Best: batch 8 at concurrency 4, 340.0 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 64 | 255.6 | 31.23 | 32.77 | 0.905 | tokenize 4.81, encode 26.61 | 1172 | 7740 |  | 7.24e-08 (1e-05) | 0 | pass |
| 8 | 4 | 1 | 88 | 340.0 | 94.06 | 104.85 | 0.907 | tokenize 7.39, encode 86.37 | 1182 | 7756 |  | 7.24e-08 (1e-05) | 0 | pass |
| 8 | 16 | 1 | 95 | 331.6 | 373.24 | 519.77 | 0.906 | tokenize 15.25, encode 359.53 | 1211 | 7852 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 33 | 257.6 | 62.08 | 64.11 | 0.905 | tokenize 9.27, encode 52.84 | 1216 | 7836 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 45 | 338.3 | 184.97 | 205.00 | 0.906 | tokenize 15.17, encode 170.58 | 1206 | 7756 |  | 7.24e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 48 | 322.0 | 750.24 | 934.13 | 0.905 | tokenize 35.07, encode 716.14 | 1130 | 7852 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 17 | 258.3 | 123.86 | n/a | 0.905 | tokenize 17.68, encode 106.23 | 1095 | 7836 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 24 | 335.3 | 371.16 | 406.16 | 0.905 | tokenize 25.32, encode 343.22 | 1078 | 7756 |  | 7.24e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 31 | 312.1 | 1460.08 | 1794.77 | 0.904 | tokenize 231.67, encode 1316.54 | 1085 | 7852 |  | 7.24e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 10 | 257.3 | 247.73 | n/a | 0.905 | tokenize 34.12, encode 213.19 | 1070 | 7772 |  | 7.24e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 12 | 326.6 | 767.84 | n/a | 0.905 | tokenize 79.97, encode 693.57 | 1079 | 7756 |  | 7.24e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 16 | 277.6 | 3429.18 | n/a | 0.905 | tokenize 629.91, encode 2824.38 | 1105 | 8876 |  | 7.24e-08 (1e-05) | 0 | pass |

### mixed: items/s (batch x concurrency)

| batch | c=1 | c=4 | c=16 |
| ---: | ---: | ---: | ---: |
| 8 | 387.2 | 470.0 | 461.0 |
| 16 | 385.8 | 481.5 | 464.4 |
| 32 | 388.2 | 477.9 | 456.7 |
| 64 | 379.6 | 461.5 | 431.2 |

Best: batch 16 at concurrency 4, 481.5 items/s.

| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB | peak VRAM MiB | micro rows / queue ms | max diff (tol) | mismatches | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| 8 | 1 | 1 | 97 | 387.2 | 23.10 | 29.55 | 0.420 | tokenize 1.66, encode 21.88 | 778 | 1376 |  | 9.31e-08 (1e-05) | 0 | pass |
| 8 | 4 | 1 | 122 | 470.0 | 68.27 | 82.63 | 0.419 | tokenize 2.55, encode 65.37 | 830 | 2476 |  | 9.31e-08 (1e-05) | 0 | pass |
| 8 | 16 | 1 | 127 | 461.0 | 262.97 | 500.25 | 0.421 | tokenize 4.02, encode 257.51 | 1009 | 4780 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 1 | 1 | 49 | 385.8 | 40.08 | 53.04 | 0.418 | tokenize 3.61, encode 36.09 | 1010 | 4716 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 4 | 1 | 63 | 481.5 | 129.03 | 144.55 | 0.418 | tokenize 5.44, encode 123.65 | 1040 | 4684 |  | 9.31e-08 (1e-05) | 0 | pass |
| 16 | 16 | 1 | 64 | 464.4 | 529.17 | 647.02 | 0.419 | tokenize 11.07, encode 521.52 | 1097 | 5804 |  | 9.31e-08 (1e-05) | 0 | pass |
| 32 | 1 | 1 | 25 | 388.2 | 81.53 | 88.13 | 0.419 | tokenize 7.09, encode 75.48 | 1091 | 5692 |  | 9.31e-08 (1e-05) | 0 | pass |
| 32 | 4 | 1 | 32 | 477.9 | 267.61 | 282.10 | 0.419 | tokenize 13.81, encode 252.70 | 1083 | 5708 |  | 9.31e-08 (1e-05) | 0 | pass |
| 32 | 16 | 1 | 32 | 456.7 | 1092.04 | 1277.87 | 0.419 | tokenize 32.79, encode 1041.48 | 1137 | 6828 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 1 | 1 | 12 | 379.6 | 171.37 | n/a | 0.408 | tokenize 13.73, encode 154.90 | 1137 | 6740 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 4 | 1 | 16 | 461.5 | 552.39 | n/a | 0.408 | tokenize 27.62, encode 514.24 | 1143 | 6732 |  | 9.31e-08 (1e-05) | 0 | pass |
| 64 | 16 | 1 | 16 | 431.2 | 2330.87 | n/a | 0.408 | tokenize 153.28, encode 2176.42 | 1171 | 7852 |  | 9.31e-08 (1e-05) | 0 | pass |


## Batching configurations (items/s)

Best valid cell of each configuration, and single-item callers (batch 1) at each concurrency. A micro-batched configuration is adopted only if its batch-1 rows beat the baseline's batch-1 rows at concurrency 4 and 16.

### embeddinggemma-2 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · long

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 75.2 | (8, 4) | 0.907 | 420.94 | 48.4 | 67.4 | 69.3 |
| baseline (max_batch_size 8) | 74.9 | (16, 4) | 0.908 | 849.44 |  |  |  |
| baseline (max_batch_size default) | 64.7 | (64, 1) | 0.900 | 991.95 |  |  |  |
| micro-w0-r8 | 64.4 | (1, 16) | 0.909 | 248.14 | 47.8 | 56.8 | 64.4 |
| micro-w2-r16 | 65.4 | (1, 16) | 0.909 | 245.18 | 44.0 | 59.4 | 65.4 |
| sorted (max_batch_size 8) | 79.4 | (64, 4) | 0.987 | 3190.09 |  |  |  |
| sorted+tokens4096 | 80.0 | (64, 16) | 0.983 | 12658.57 |  |  |  |
| tokens4096 | 75.2 | (16, 4) | 0.908 | 841.24 |  |  |  |

### embeddinggemma-2 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · mixed

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 142.3 | (1, 16) | 1.000 | 111.30 | 64.9 | 131.2 | 142.3 |
| baseline (max_batch_size 8) | 84.0 | (16, 4) | 0.391 | 753.57 |  |  |  |
| baseline (max_batch_size default) | 72.6 | (8, 1) | 0.416 | 106.38 |  |  |  |
| micro-w0-r8 | 74.9 | (1, 4) | 0.586 | 52.86 | 63.1 | 74.9 | 66.6 |
| micro-w2-r16 | 77.5 | (1, 4) | 0.587 | 53.76 | 58.2 | 77.5 | 66.8 |
| sorted (max_batch_size 8) | 174.9 | (64, 16) | 0.850 | 5764.22 |  |  |  |
| sorted+tokens4096 | 153.1 | (64, 16) | 0.712 | 6012.76 |  |  |  |
| tokens4096 | 87.9 | (16, 4) | 0.407 | 687.06 |  |  |  |

### embeddinggemma-2 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · short

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1279.1 | (64, 4) | 0.644 | 199.43 | 75.7 | 173.4 | 202.6 |
| micro-w0-r8 | 575.9 | (1, 16) | 0.750 | 27.53 | 70.3 | 141.2 | 575.9 |
| micro-w2-r16 | 557.2 | (1, 16) | 0.737 | 28.96 | 64.2 | 166.7 | 557.2 |

### gliner2-base-v1 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · long

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 76.4 | (64, 4) | 0.963 | 3315.88 | 60.7 | 60.1 | 59.4 |
| baseline (max_batch_size 8) | 72.3 | (64, 4) | 0.967 | 3077.46 |  |  |  |
| baseline (max_batch_size default) | 71.3 | (16, 1) | 0.965 | 225.54 |  |  |  |
| micro-w0-r8 | 72.9 | (1, 16) | 0.966 | 219.89 | 59.6 | 68.3 | 72.9 |
| micro-w2-r16 | 74.3 | (1, 16) | 0.966 | 215.04 | 54.1 | 68.5 | 74.3 |
| sorted (max_batch_size 8) | 73.6 | (64, 4) | 0.992 | 3025.08 |  |  |  |
| sorted+tokens4096 | 73.4 | (64, 4) | 0.992 | 2745.31 |  |  |  |
| tokens4096 | 72.3 | (8, 4) | 0.968 | 440.00 |  |  |  |

### gliner2-base-v1 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · mixed

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 147.3 | (1, 1) | 1.000 | 4.73 | 147.3 | 104.8 | 105.6 |
| baseline (max_batch_size 8) | 110.4 | (16, 1) | 0.400 | 132.69 |  |  |  |
| baseline (max_batch_size default) | 111.2 | (8, 1) | 0.399 | 96.12 |  |  |  |
| micro-w0-r8 | 138.9 | (1, 4) | 0.630 | 19.75 | 138.4 | 138.9 | 98.5 |
| micro-w2-r16 | 113.9 | (1, 4) | 0.474 | 45.08 | 112.5 | 113.9 | 97.6 |
| sorted (max_batch_size 8) | 214.3 | (64, 4) | 0.819 | 1023.27 |  |  |  |
| sorted+tokens4096 | 222.0 | (64, 16) | 0.775 | 3760.47 |  |  |  |
| tokens4096 | 110.6 | (8, 1) | 0.399 | 96.84 |  |  |  |

### gliner2-base-v1 · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · short

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 602.6 | (64, 4) | 0.579 | 422.31 | 227.1 | 129.0 | 129.6 |
| micro-w0-r8 | 560.5 | (1, 16) | 0.687 | 27.64 | 225.3 | 373.4 | 560.5 |
| micro-w2-r16 | 567.5 | (1, 16) | 0.679 | 27.23 | 147.0 | 405.0 | 567.5 |

### multilingual-e5-small · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · long

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 350.9 | (1, 16) | 1.000 | 44.61 | 189.9 | 316.1 | 348.7 |
| baseline (max_batch_size 8) | 337.5 | (16, 4) | 0.906 | 183.35 |  |  |  |
| baseline (max_batch_size default) | 269.0 | (32, 1) | 0.887 | 119.01 |  |  |  |
| micro-w0-r8 | 297.6 | (1, 16) | 0.908 | 54.05 | 166.0 | 261.5 | 297.6 |
| micro-w2-r16 | 296.8 | (1, 16) | 0.904 | 53.69 | 121.2 | 261.1 | 296.8 |
| sorted (max_batch_size 8) | 363.6 | (32, 4) | 0.968 | 350.99 |  |  |  |
| sorted+tokens4096 | 354.6 | (16, 4) | 0.948 | 177.71 |  |  |  |
| tokens4096 | 340.0 | (8, 4) | 0.907 | 94.06 |  |  |  |

### multilingual-e5-small · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · mixed

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 857.6 | (1, 16) | 1.000 | 17.97 | 330.9 | 730.8 | 857.6 |
| baseline (max_batch_size 8) | 481.8 | (16, 4) | 0.419 | 132.58 |  |  |  |
| baseline (max_batch_size default) | 387.4 | (8, 1) | 0.420 | 23.25 |  |  |  |
| micro-w0-r8 | 427.5 | (1, 4) | 0.634 | 8.96 | 289.9 | 427.5 | 383.8 |
| micro-w2-r16 | 409.7 | (1, 4) | 0.581 | 9.03 | 166.9 | 409.7 | 399.4 |
| sorted (max_batch_size 8) | 986.3 | (64, 4) | 0.855 | 253.76 |  |  |  |
| sorted+tokens4096 | 874.2 | (64, 4) | 0.702 | 288.39 |  |  |  |
| tokens4096 | 481.5 | (16, 4) | 0.418 | 129.03 |  |  |  |

### multilingual-e5-small · cuda · intra-op 4 · NVIDIA GeForce RTX 3090 · short

| config | best items/s | at (batch, conc) | padding eff | p50 ms | b1 c=1 | b1 c=4 | b1 c=16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8507.4 | (64, 4) | 0.548 | 29.72 | 465.1 | 1318.9 | 1937.5 |
| micro-w0-r8 | 3191.5 | (1, 16) | 0.711 | 4.90 | 404.1 | 786.3 | 3191.5 |
| micro-w2-r16 | 3902.0 | (1, 16) | 0.656 | 3.79 | 213.3 | 831.6 | 3902.0 |
