| condition | n | correct tool | wrong tool | invalid name | malformed | no call | args valid (of calls to offered tools) | labelled args match (of correct, labelled cases) |
|---|---|---|---|---|---|---|---|---|
| free-all | 100 | 88 | 8 | 0 | 1 | 3 | 96/96 | 63/66 |
| guided-all | 100 | 91 | 8 | 0 | 1 | 0 | 99/99 | 65/67 |
| free-subset | 100 | 83 | 2 | 0 | 0 | 15 | 85/85 | 60/64 |
| guided-subset | 100 | 95 | 5 | 0 | 0 | 0 | 100/100 | 67/70 |

Invalid names (condition: case id -> name):
  free-all: [(23, 'create_note'), (35, None), (60, None), (70, None)]
  guided-all: [(23, 'create_note')]
  free-subset: [(4, None), (12, None), (16, None), (19, None), (25, None), (30, None), (35, None), (48, None), (60, None), (68, None), (69, None), (70, None), (72, None), (73, None), (74, None)]
  guided-subset: []

| condition | TTFT p50 | TTFT p90 | total p50 | total p90 | completion tokens p50 | decode tok/s p50 | decode tok/s p90 | ms/token p50 |
|---|---|---|---|---|---|---|---|---|
| free-all | 353.3 | 520.4 | 1419.0 | 2163.2 | 39 | 38.6 | 39.3 | 37.81 |
| guided-all | 341.7 | 522.4 | 1517.6 | 2199.5 | 39 | 35.3 | 37.3 | 38.47 |
| free-subset | 158.8 | 219.6 | 944.3 | 1624.7 | 39 | 51.7 | 54.0 | 23.78 |
| guided-subset | 158.7 | 219.3 | 1048.3 | 1573.1 | 40 | 47.0 | 50.6 | 25.41 |

Paired ratios (B / A (same case, same tools), p50 and p90 of per-case ratios:
  guided-all / free-all ttft: p50 0.983 p90 1.069
  guided-all / free-all total: p50 1.060 p90 1.155
  guided-all / free-all decode tok/s: p50 0.930 p90 0.997
  guided-all / free-all total, cases with equal token counts (n=59): p50 1.053 p90 1.135
  identical text: 93/100
  guided-subset / free-subset ttft: p50 0.976 p90 1.069
  guided-subset / free-subset total: p50 1.085 p90 1.205
  guided-subset / free-subset decode tok/s: p50 0.916 p90 1.013
  guided-subset / free-subset total, cases with equal token counts (n=35): p50 1.080 p90 1.160
  identical text: 78/100

free-all on 'tempting' prompts: Counter({'correct': 37, 'wrong_tool': 8, 'no_call': 3})
