| Load | Policy | Goodput (x capacity) | SLO met | p50 TTFT | p99 TTFT | p50 latency | p99 latency | Longest 10%: p50 latency |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5x | FCFS | 0.47 | 100% | 0.07 s | 0.18 s | 1.00 s | 5.56 s | 3.31 s |
| 0.5x | Shortest feasible job first | 0.48 | 100% | 0.06 s | 0.14 s | 0.85 s | 4.48 s | 2.88 s |
| 1x | FCFS | 0.86 | 100% | 0.08 s | 0.21 s | 1.78 s | 9.05 s | 6.59 s |
| 1x | Shortest feasible job first | 0.90 | 100% | 0.08 s | 0.26 s | 1.64 s | 9.35 s | 7.00 s |
| 2x | FCFS | 0.95 | 89% | 0.82 s | 3.62 s | 5.07 s | 15.68 s | 13.05 s |
| 2x | Shortest feasible job first | 0.98 | 92% | 0.62 s | 6.39 s | 4.46 s | 15.51 s | 12.78 s |
| 3x | FCFS | 1.45 | 98% | 0.77 s | 2.96 s | 3.85 s | 11.31 s | 9.37 s |
| 3x | Shortest feasible job first | 1.17 | 88% | 0.95 s | 5.72 s | 4.09 s | 12.66 s | 10.52 s |
| 4x | FCFS | 0.36 | 26% | 7.88 s | 15.88 s | 12.91 s | 21.31 s | 18.02 s |
| 4x | Shortest feasible job first | 0.73 | 58% | 2.78 s | 17.05 s | 6.91 s | 25.73 s | 22.29 s |
