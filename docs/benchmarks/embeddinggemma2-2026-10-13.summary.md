Medians with 95% hierarchical bootstrap CIs (runs, then samples). p5/p95 need n >= 20; p99 needs n >= 1000 and is otherwise not reported.

### C# (library) vs Python, same backend

| Provider | Mode | Path | Workload | Conc. | C# median ms | Python median ms | C#/Python | 95% CI | Delta ms (95% CI) | n C#/Py |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |

### All series

| Provider | Mode | Path | Workload | Conc. | Harness | Runs | n | Median ms (95% CI) | p5–p95 ms | p99 ms | Items/s | Stage medians ms | Quality max abs |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: |
| cpu | cold | text | cold-start | 1 | csharp/library | 1 | 3 | 5,686 (5,230–5,723) | n/a | n/a | 0.2 | spawn_to_main 90.0, fixture_read 257, tokenizer_load 1,229, session_create 3,617, tokenize 89.2, encode 322, main_to_result 5,606, process_wall 6,049 | pass |
| cpu | warm | image | huge-b1 | 1 | csharp/library | 1 | 2 | 16,518 (16,192–16,844) | n/a | n/a | 0.1 | decode 220, embed 16,298 | 3.43e-07 |
| cpu | warm | image | photo-b1 | 1 | csharp/library | 1 | 4 | 16,059 (13,136–16,626) | n/a | n/a | 0.1 | decode 9.01, embed 16,050 | 6.60e-07 |
| cpu | warm | image | photo-b4 | 1 | csharp/library | 1 | 1 | 59,460 (59,460–59,460) | n/a | n/a | 0.1 | decode 67.1, embed 59,393 | 1.32e-06 |
| cpu | warm | image | preprocess-only | 1 | csharp/library | 1 | 10 | 58.1 (34.1–235) | n/a | n/a | 7.9 | decode 5.35, preprocess 21.4 | pass |
| cpu | warm | image | tiny-b1 | 1 | csharp/library | 1 | 4 | 15,980 (15,499–16,269) | n/a | n/a | 0.1 | decode 0.34, embed 15,980 | 2.31e-07 |
| cpu | warm | image | two-images-interleaved | 1 | csharp/library | 1 | 2 | 12,524 (11,586–13,462) | n/a | n/a | 0.1 | decode 12.8, embed 12,511 | 6.79e-07 |
| cpu | warm | text | long-b1 | 1 | csharp/library | 1 | 2 | 8,985 (8,186–9,784) | n/a | n/a | 0.1 | tokenize 3.46, encode 8,982 | 1.56e-07 |
| cpu | warm | text | long-b4 | 1 | csharp/library | 1 | 1 | 27,436 (27,436–27,436) | n/a | n/a | 0.1 | tokenize 9.27, encode 27,426 | 1.94e-07 |
| cpu | warm | text | short-b1 | 1 | csharp/library | 1 | 25 | 299 (226–379) | 170–519 | n/a | 3.1 | tokenize 0.08, encode 298 | 1.97e-07 |
| cpu | warm | text | short-b16 | 1 | csharp/library | 1 | 5 | 4,011 (3,425–4,431) | n/a | n/a | 4.1 | tokenize 3.91, encode 4,006 | 1.94e-07 |
| cpu | warm | text | short-b8 | 1 | csharp/library | 1 | 10 | 2,036 (1,242–2,089) | n/a | n/a | 4.4 | tokenize 0.28, encode 2,035 | 1.94e-07 |
| cpu | warm | text | words100-b1 | 1 | csharp/library | 1 | 8 | 603 (577–750) | n/a | n/a | 1.5 | tokenize 0.32, encode 602 | 0.00e+00 |
| cpu | warm | text | words400-b1 | 1 | csharp/library | 1 | 4 | 2,664 (2,446–3,206) | n/a | n/a | 0.4 | tokenize 1.29, encode 2,663 | 0.00e+00 |

### Runs

| Started (UTC) | Harness | Provider | Mode | Threads | Quality (ids/text max abs, pass) | Load before | GPUs before |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13T17:39:37 | csharp/library | cpu | warm | 6 | 138 cases, True | [15.7, 18.8, 21.19] | GPU0 93% 10995 MiB, GPU1 0% 729 MiB |
| 2026-10-13T17:47:19 | csharp/library | cpu | cold | 6 | 1.97e-07 (first request), True | [20.49, 19.97, 21] | GPU0 96% 10997 MiB, GPU1 17% 725 MiB |
| 2026-10-13T17:47:43 | csharp/library | cpu | warm | 6 | 8 cases, True | [19.41, 19.75, 20.9] | GPU0 100% 10997 MiB, GPU1 0% 737 MiB |
