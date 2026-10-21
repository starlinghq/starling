| condition | n | correct tool | wrong tool | invalid name | malformed | no call | args valid (of calls to offered tools) | labelled args match (of correct, labelled cases) |
|---|---|---|---|---|---|---|---|---|
| free-all | 100 | 88 | 8 | 0 | 1 | 3 | 96/96 | 63/66 |
| free-all-repeat | 100 | 88 | 8 | 0 | 1 | 3 | 96/96 | 63/66 |
| free-subset | 100 | 83 | 2 | 0 | 0 | 15 | 85/85 | 60/64 |
| free-subset-repeat | 100 | 83 | 2 | 0 | 0 | 15 | 85/85 | 60/64 |

Invalid names (condition: case id -> name):
  free-all: [(23, 'create_note'), (35, None), (60, None), (70, None)]
  free-all-repeat: [(23, 'create_note'), (35, None), (60, None), (70, None)]
  free-subset: [(4, None), (12, None), (16, None), (19, None), (25, None), (30, None), (35, None), (48, None), (60, None), (68, None), (69, None), (70, None), (72, None), (73, None), (74, None)]
  free-subset-repeat: [(4, None), (12, None), (16, None), (19, None), (25, None), (30, None), (35, None), (48, None), (60, None), (68, None), (69, None), (70, None), (72, None), (73, None), (74, None)]

| condition | TTFT p50 | TTFT p90 | total p50 | total p90 | completion tokens p50 | decode tok/s p50 | decode tok/s p90 | ms/token p50 |
|---|---|---|---|---|---|---|---|---|
| free-all | 340.5 | 511.4 | 1339.3 | 2297.1 | 39 | 38.5 | 39.2 | 35.40 |
| free-all-repeat | 341.7 | 496.2 | 1333.5 | 2336.3 | 39 | 38.7 | 39.2 | 34.58 |
| free-subset | 156.5 | 210.1 | 901.0 | 1557.2 | 39 | 52.2 | 53.8 | 23.16 |
| free-subset-repeat | 159.1 | 204.7 | 906.8 | 1631.6 | 39 | 52.0 | 53.7 | 23.19 |

Paired ratios (B / A (same case, same tools), p50 and p90 of per-case ratios:
  free-all-repeat / free-all ttft: p50 1.003 p90 1.097
  free-all-repeat / free-all total: p50 1.000 p90 1.032
  free-all-repeat / free-all decode tok/s: p50 1.002 p90 1.056
  free-all-repeat / free-all total, cases with equal token counts (n=100): p50 1.000 p90 1.032
  identical text: 100/100
  free-subset-repeat / free-subset ttft: p50 0.998 p90 1.098
  free-subset-repeat / free-subset total: p50 1.000 p90 1.037
  free-subset-repeat / free-subset decode tok/s: p50 0.999 p90 1.055
  free-subset-repeat / free-subset total, cases with equal token counts (n=100): p50 1.000 p90 1.037
  identical text: 100/100

free-all on 'tempting' prompts: Counter({'correct': 37, 'wrong_tool': 8, 'no_call': 3})
